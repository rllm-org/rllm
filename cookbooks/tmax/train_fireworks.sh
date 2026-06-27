#!/usr/bin/env bash
# Train a Tmax-style terminal agent on tmax-15k, eval on Terminal-Bench —
# Fireworks backend (managed, LoRA). This is the ACCESSIBLE variant: it trains a
# LoRA adapter on Qwen3.5-9B instead of the full-parameter DPPO run Tmax used.
# For the faithful full fine-tune, use train_verl.sh.
#
# Prerequisites:
#   1. Install rllm with fireworks + harbor extras:  uv pip install -e ".[fireworks,harbor]"
#   2. Install this cookbook:                        uv pip install --no-deps -e cookbooks/tmax
#   3. Pull the datasets:                            python cookbooks/tmax/prepare_data.py
#   4. Set your API key:                             export FIREWORKS_API_KEY=...
#
# The trainer job and inference deployment are provisioned on Fireworks at
# startup and torn down on shutdown.
#
# This is aligned field-for-field with Tmax's official DPPO run
# (scripts/tmax/RL/qwen35_9b.sh); the ONLY intended differences are those LoRA
# and the rLLM backend force (listed at the bottom). Matched values:
#   - Context: max_length=67584 = Tmax pack_length (51200 cumulative prompt
#     window + 16384 per-turn response = Tmax's response_length 65536 + initial
#     prompt 2048). Cumulative token mode forwards exact trajectory tokens.
#   - per-turn response 16384 = per_turn_max_tokens; 64 tool calls = max_steps.
#   - In FULLY-ASYNC mode, prompts/step = async_training.mini_batch_size (the
#     dataloader batch is forced to 1; data.train_batch_size is IGNORED). So
#     mini_batch_size=8 = num_unique_prompts_rollout, x group 32 (GROUP_SIZE) =
#     256 rollouts/step = num_unique_prompts_rollout x num_samples_per_prompt_rollout.
#   - async_steps=4 -> staleness_threshold=3.0: max in-flight groups =
#     (1+staleness)*trigger*mini_batch, so (1+3)*1 = 4 steps ahead = async_steps.
#   - ~500 optimizer steps (total_batches; 500*256 = Tmax's total_episodes
#     128000), 1 epoch, save_freq 20, seed 42, temperature 1.0, constant LR.
#   - Centered advantages (norm_adv_by_std_in_grpo=false) = Tmax
#     advantage_normalization_type=centered. No KL (beta=0.0). Compact filtering
#     drops overlong rollouts (Tmax's overlong handling).
#
# Intended differences (LoRA + backend, NOT recipe choices):
#   - LoRA-32 adapter, not full-parameter DPPO. LoRA needs a higher LR than the
#     paper's full-FT 1e-6, so this uses 2e-5 (1e-6 barely moves an adapter).
#   - loss_fn: Fireworks DPPO custom forward/backward, using rollout logprobs
#     as the behavior anchor with a TV trust region. Centered advantages, no-KL,
#     and outcome-only rewards are matched.
#   - lm_head_fp32 / Liger GRPO are backend-internal (not exposed on Fireworks).
#
# Model: Qwen3.5-9B + LoRA-32 on the qwen3p5-9b-256k-lora training shape.
# Sandbox backend via TERMINAL_SANDBOX_BACKEND; harness via TMAX_HARNESS.
#
# Flip GRPO -> ECHO (free dense supervision from terminal output; great for a
# hard, failure-heavy benchmark) by appending: rllm.algorithm.adv_estimator=echo
#
# Defaults are Tmax-aligned (GROUP_SIZE=32, mini_batch_size=8 prompts/step,
# 67584 ctx). Scale down via env vars to cut cost: TRAINER_REPLICAS (trainer
# copies, default 1), ROLLOUT_REPLICAS (inference copies for rollout throughput,
# default 6), GROUP_SIZE (samples/prompt, default 32). Override anything else
# with Hydra args. Examples:
#   bash train_fireworks.sh                                          # Tmax-aligned
#   ROLLOUT_REPLICAS=2 GROUP_SIZE=8 bash train_fireworks.sh \
#       rllm.async_training.mini_batch_size=2                        # cheap smoke

set -euo pipefail

export TERMINAL_SANDBOX_BACKEND="${TERMINAL_SANDBOX_BACKEND:-daytona}"
export TMAX_HARNESS="${TMAX_HARNESS:-mini-swe-agent}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-1800}"
export RLLM_SANDBOX_TIMEOUT_S="${RLLM_SANDBOX_TIMEOUT_S:-2400}"

# --- Fireworks scale knobs (the two distinct "replicas") -------------------
#  TRAINER_REPLICAS  -> fireworks_config.policy_trainer_replica_count
#      Copies of the LoRA *trainer* job (fwd/bwd + optim on the training shape).
#      The gradient side; rarely the RL bottleneck. 1 is standard for LoRA-9B.
#  ROLLOUT_REPLICAS  -> fireworks_config.rollout_deployment_replica_count
#      Copies of the *inference* deployment that generates rollouts. This is the
#      throughput knob (analog of Tmax's 6 inference nodes / 48 vLLM engines) and
#      seeds the adaptive sampling window (8 * replica_count). Raise for speed/$$.
#  GROUP_SIZE        -> samples per prompt. Default 32 = Tmax's
#      num_samples_per_prompt_rollout. Drop to 16/8 to cut cost (lower fidelity).
#  Reference-trainer replicas stay 0: Tmax has beta=0 (no KL), so no ref model.
TRAINER_REPLICAS="${TRAINER_REPLICAS:-1}"
ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-4}"
GROUP_SIZE="${GROUP_SIZE:-32}"

python -u train.py \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/qwen3p5-9b \
    model.tokenizer_model=Qwen/Qwen3.5-9B \
    fireworks_config.policy_trainer_shape_id=accounts/fireworks/trainingShapes/qwen3p5-9b-256k \
    fireworks_config.policy_trainer_replica_count=$TRAINER_REPLICAS \
    fireworks_config.rollout_deployment_replica_count=$ROLLOUT_REPLICAS \
    fireworks_config.reference_trainer_replica_count=0 \
    fireworks_infra.deployments.rollout.disable_speculative_decoding=true \
    concurrency=null \
    training.group_size=$GROUP_SIZE \
    training.learning_rate=1e-6 \
    training.max_length=67584 \
    rllm.data.max_prompt_length=67584 \
    rllm.data.max_response_length=16384 \
    rllm.data.train_batch_size=1 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=false \
    rllm.algorithm.loss_fn=dppo \
    rllm.algorithm.dppo_divergence_type=tv \
    rllm.algorithm.dppo_divergence_threshold=0.1 \
    rllm.algorithm.lr_schedule=constant \
    rllm.algorithm.kl_beta=0.0 \
    rllm.algorithm.rollout_correction.bypass_mode=true \
    rllm.async_training.enable=true \
    rllm.async_training.mini_batch_size=8 \
    rllm.async_training.fwd_bwd_group_size=1 \
    rllm.async_training.staleness_threshold=4.0 \
    rllm.async_training.trigger_parameter_sync_step=1 \
    rllm.async_training.partial_rollout=true \
    rllm.compact_filtering.enable=false \
    rllm.rejection_sample.filter_uniform_groups=true \
    rllm.workflow.n_parallel_tasks=512 \
    rllm.workflow.raise_on_error=false \
    rllm.gateway.port=9090 \
    rllm.gateway.cumulative_token_mode=true \
    rllm.gateway.renderer_family=qwen3.5 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.total_batches=500 \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='tmax' \
    rllm.trainer.experiment_name='tmax-qwen3p5-9b-fireworks-dppo' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=-1 \
    rllm.trainer.save_freq=10 \
    "$@"

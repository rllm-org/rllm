#!/usr/bin/env bash
# Reproduce Ai2's Tmax-9B terminal-agent RL run with the verl (distributed)
# backend — FULL fine-tuning of Qwen3.5-9B, the faithful path.
#
# Prerequisites:
#   1. Install rllm with verl + harbor extras:  uv pip install -e ".[verl,harbor]"
#   2. Install megatron:                         bash scripts/install_megatron.sh <cu128|cu129|...>
#   3. Install this cookbook:                    uv pip install --no-deps -e cookbooks/tmax
#   4. Pull the datasets:                        python cookbooks/tmax/prepare_data.py
#
# ---------------------------------------------------------------------------
# Mapping Tmax's official DPPO recipe (training/open-instruct/scripts/tmax/RL/
# qwen35_9b.sh, open_instruct/grpo_fast.py) onto rLLM's unified trainer:
#
#   Tmax (open-instruct)              | here (rLLM / verl)
#   ----------------------------------|------------------------------------------
#   model hamishivi/Qwen3.5-9B        | Qwen/Qwen3.5-9B, FULL fine-tune (no LoRA)
#   num_unique_prompts_rollout 8      | data.train_batch_size=8
#   num_samples_per_prompt_rollout 32 | actor_rollout_ref.rollout.n=32  (group size)
#   response_length 65536 (episode)   | per-turn windows summing to ~64K (below)
#   per_turn_max_tokens 16384         | data.max_response_length=16384
#   max_steps 64 (tool calls/episode) | TERMINUS_MAX_TURNS=64
#   learning_rate 1e-6                | actor.optim.lr=1e-6
#   lr_scheduler_type constant        | algorithm.lr_schedule=constant
#   beta 0.0 (no KL)                  | use_kl_loss=False / kl_beta=0.0
#   temperature 1.0                   | rollout.temperature=1.0
#   advantage_normalization centered  | norm_adv_by_std_in_grpo=false (mean-center)
#   loss_fn dppo (tv, thr 0.1)        | adv_estimator=grpo + PPO clip  (see NOTE)
#   verification_reward 1.0           | per-task tests/test.sh -> reward 1.0/0.0
#   seed 42                           | data.seed=42
#   deepspeed_stage 3 (full param)    | FSDP, param+optimizer offload
#
# NOTE — DPPO: rLLM does not implement Tmax's exact DPPO loss (a total-variation
# trust region, divergence threshold 0.1). We map it to GRPO with PPO-style ratio
# clipping, which is the practical analog (no learned reward model, outcome-only
# advantages, on-policy clip as the trust region). The `centered` advantage
# normalization IS reproduced (norm_adv_by_std_in_grpo=false). FP32 LM head and
# Liger GRPO loss (lm_head_fp32 / use_liger_grpo_loss in their script) are
# backend-internal and not exposed here. See README "Fidelity vs. the paper".
#
# HARDWARE — Tmax used 8x H100 nodes (2 for training w/ sequence_parallel_size 4,
# 6 for vLLM inference). rLLM's verl path colocates rollout + training
# (hybrid_engine), a different topology. Full FT of a 9B at 65K context + group
# 32 is heavy: a single 8x H100 node is the practical floor and you will likely
# want >=2 nodes and/or Ulysses sequence parallelism (append
# actor_rollout_ref.actor.ulysses_sequence_parallel_size=4). Set NNODES /
# GPUS_PER_NODE below. For an accessible (LoRA) run, use train_fireworks.sh or
# train_tinker.sh instead.
#
# Sandbox backend is chosen by TERMINAL_SANDBOX_BACKEND (docker | local | modal |
# daytona; default modal). Harness via TMAX_HARNESS (terminus2 | mini-swe-agent).
#
# Override anything by appending Hydra args:
#   bash train_verl.sh actor_rollout_ref.rollout.n=16 trainer.nnodes=2

set -euo pipefail

unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

export TERMINAL_SANDBOX_BACKEND="${TERMINAL_SANDBOX_BACKEND:-modal}"
export TMAX_HARNESS="${TMAX_HARNESS:-terminus2}"
# Per-rollout tool-call cap (Tmax max_steps=64). Read by train.py.
export TERMINUS_MAX_TURNS="${TERMINUS_MAX_TURNS:-64}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-1800}"
# Provider-agnostic sandbox LIFETIME floor (seconds). Must exceed the agent run
# timeout above plus setup/verify, or sandboxes get reaped mid-rollout —
# surfacing as "Sandbox has already shut down" (Modal) / ENOSPC mid-run. Honored
# by every backend (Modal hard timeout; Daytona idle auto-stop, in minutes).
export RLLM_SANDBOX_TIMEOUT_S="${RLLM_SANDBOX_TIMEOUT_S:-2400}"

NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"

# NOTE: use the rllm.algorithm.* namespace (preferred) — verl-native algorithm.*
# overrides warn as deprecated. rllm.algorithm.kl_beta maps to verl's
# kl_loss_coef and derives actor.use_kl_loss. Tmax's constant LR is verl's
# default (no warmup/decay configured). rllm.data.seed sets the data seed
# (verl's top-level data config has no seed key).
python -u train.py \
    rllm/backend=verl \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=false \
    rllm.algorithm.use_rllm=true \
    rllm.algorithm.kl_beta=0.0 \
    rllm.algorithm.warmup_steps_ratio=0.0 \
    data.train_batch_size=8 \
    data.val_batch_size=-1 \
    data.max_prompt_length=49152 \
    data.max_response_length=16384 \
    rllm.data.seed=42 \
    +model.name=$MODEL_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    +actor_rollout_ref.rollout.max_model_len=65536 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    rllm.workflow.n_parallel_tasks=64 \
    rllm.workflow.raise_on_error=false \
    rllm.gateway.port=9091 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=tmax \
    trainer.experiment_name=tmax-qwen3p5-9b-dppo-verl \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=disable \
    "$@"

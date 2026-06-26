#!/usr/bin/env bash
# Train a Tmax-style terminal agent on tmax-15k, eval on Terminal-Bench —
# Tinker backend (single-machine, LoRA). Accessible variant: a LoRA adapter on
# Qwen3.5-9B rather than the full-parameter DPPO run. For the faithful full
# fine-tune, use train_verl.sh.
#
# Prerequisites:
#   1. Install rllm with tinker + harbor extras:  uv pip install -e ".[tinker,harbor]"
#   2. Install this cookbook:                     uv pip install --no-deps -e cookbooks/tmax
#   3. Pull the datasets:                         python cookbooks/tmax/prepare_data.py
#
# What this configures:
#   - Async GRPO with compact filtering.
#   - Centered advantages (norm_adv_by_std_in_grpo=false) = Tmax's
#     advantage_normalization_type=centered. No KL (beta=0.0). Temperature 1.0.
#   - 64 tool-call cap per episode (Tmax max_steps=64).
#   - 64K single-turn window (49K prompt history + 16K response), matching
#     Tmax's per_turn_max_tokens=16384. Tmax found ~65K output length important
#     for 9B stability — keep max_length high.
#
# LoRA deviations from the paper: LoRA-32 not full-FT (so LR 2e-5, not 1e-6);
# group_size 8 not 32 (raise it if you have the GPUs/time).
#
# Sandbox backend via TERMINAL_SANDBOX_BACKEND; harness via TMAX_HARNESS.
# Flip to ECHO with: rllm.algorithm.adv_estimator=echo
#
# Override anything by appending Hydra args:
#   bash train_tinker.sh training.group_size=16

set -euo pipefail

export TERMINAL_SANDBOX_BACKEND="${TERMINAL_SANDBOX_BACKEND:-modal}"
export TMAX_HARNESS="${TMAX_HARNESS:-terminus2}"
export TERMINUS_MAX_TURNS="${TERMINUS_MAX_TURNS:-64}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-1800}"
# Provider-agnostic sandbox lifetime floor (seconds), honored by every backend
# (Modal hard timeout; Daytona idle auto-stop, converted to minutes).
export RLLM_SANDBOX_TIMEOUT_S="${RLLM_SANDBOX_TIMEOUT_S:-2400}"

python -u train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3.5-9B \
    model.lora_rank=32 \
    training.group_size=8 \
    training.learning_rate=2e-5 \
    training.max_length=65536 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.rollout.val.temperature=1.0 \
    rllm.rollout.val.top_p=1.0 \
    data.max_prompt_length=49152 \
    data.max_response_length=16384 \
    data.train_batch_size=1 \
    data.val_batch_size=-1 \
    rllm.compact_filtering.enable=true \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=false \
    rllm.algorithm.lr_schedule=constant \
    rllm.algorithm.kl_beta=0.0 \
    rllm.async_training.enable=true \
    rllm.async_training.mini_batch_size=16 \
    rllm.async_training.fwd_bwd_group_size=1 \
    rllm.async_training.staleness_threshold=0.5 \
    rllm.async_training.trigger_parameter_sync_step=1 \
    rllm.async_training.partial_rollout=true \
    rllm.workflow.n_parallel_tasks=128 \
    rllm.workflow.raise_on_error=false \
    rllm.gateway.port=9091 \
    rllm.gateway.cumulative_token_mode=true \
    rllm.renderer.family=qwen3.5 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='tmax' \
    rllm.trainer.experiment_name='tmax-qwen3p5-9b-tinker-lora' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=10 \
    rllm.trainer.save_freq=10 \
    "$@"

#!/usr/bin/env bash
# Train an SWE agent on R2E-Gym with the Fireworks backend in SYNCHRONOUS mode.
#
# This is the simpler, on-policy variant of train_fireworks.sh (which uses
# fully-async GRPO). Each step generates a full batch of rollouts, then takes
# one optimizer step — easier to reason about for testing/debugging. Mirrors
# the synchronous fireworks recipe other cookbooks use (e.g. deepcoder).
#
# Prerequisites:
#   1. Install rllm with fireworks extras:  uv pip install -e ".[fireworks]"
#   2. Install this cookbook:                uv pip install --no-deps -e cookbooks/swe-rl
#   3. Pull the datasets:                    python cookbooks/swe-rl/prepare_data.py
#   4. Set your API key:                     export FIREWORKS_API_KEY=...
#
# The trainer job and inference deployment are provisioned on Fireworks at
# startup (via the cookbook's training.provision API) and torn down on shutdown.
#
# Differences from train_fireworks.sh:
#   - No rllm.async_training.* (async disabled → synchronous generate→train).
#   - data.train_batch_size is the real per-step task count (async forces it to 1).
#   - The effective batch is train_batch_size x group_size rollouts per step.
#
# Model: Qwen3.5-9B + LoRA-32 on the qwen3p5-9b-256k-lora training shape.
# Fireworks' public catalog ships a 3.5-9B LoRA shape but no 3.5-4B, so this
# uses the 9B the README names as the base model rather than the tinker recipe's
# 4B. To change it, swap model.name / model.tokenizer_model /
# fireworks_config.policy_trainer_shape_id together (see docs/backends/fireworks.mdx).
#
# Sandbox backend: SWE_SANDBOX_BACKEND (docker | local | modal | daytona; default
# modal). Cap the eval set with SWE_VAL_MAX. Per-rollout agent timeout:
# RLLM_HARNESS_RUN_TIMEOUT_S.
#
# Qwen3.5 is a reasoning family; if the harness can't parse reasoning output,
# disable it by appending: rollout_engine.reasoning_effort=none
#
# Override anything by passing extra Hydra args after the script:
#   bash train_fireworks_sync.sh data.train_batch_size=2 training.group_size=4

set -euo pipefail

export SWE_SANDBOX_BACKEND="${SWE_SANDBOX_BACKEND:-modal}"
export SWE_VAL_MAX="${SWE_VAL_MAX:-16}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-1800}"

python -u train.py \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/qwen3p5-9b \
    model.tokenizer_model=Qwen/Qwen3.5-9B \
    model.lora_rank=32 \
    fireworks_config.policy_trainer_shape_id=accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora \
    fireworks_config.rollout_deployment_replica_count=4 \
    training.group_size=8 \
    training.learning_rate=2e-5 \
    training.max_length=65536 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.rollout.val.temperature=1.0 \
    rllm.rollout.val.top_p=1.0 \
    data.max_prompt_length=57344 \
    data.max_response_length=8192 \
    data.train_batch_size=16 \
    data.val_batch_size=-1 \
    rllm.data.max_prompt_length=57344 \
    rllm.data.max_response_length=8192 \
    rllm.data.train_batch_size=16 \
    rllm.data.val_batch_size=-1 \
    rllm.compact_filtering.enable=true \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.workflow.n_parallel_tasks=128 \
    rllm.workflow.raise_on_error=false \
    rllm.gateway.port=9090 \
    rllm.gateway.cumulative_token_mode=true \
    rllm.gateway.renderer_family=qwen3.5 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='swe-rl' \
    rllm.trainer.experiment_name='r2egym-terminus2-qwen3p5-9b-fireworks-sync' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=10 \
    rllm.trainer.save_freq=10 \
    "$@"

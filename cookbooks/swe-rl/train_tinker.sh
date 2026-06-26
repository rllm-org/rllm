#!/usr/bin/env bash
# Train an SWE agent on R2E-Gym, eval on SWE-bench Verified.
#
# Prerequisites:
#   1. Install rllm with tinker extras:   uv pip install -e ".[tinker]"
#   2. Install this cookbook:              uv pip install --no-deps -e cookbooks/swe-rl
#   3. Pull the datasets:                  python cookbooks/swe-rl/prepare_data.py
#
# What this configures, in plain English:
#   - Async GRPO with compact filtering (drop too-long rollouts before grad).
#   - 64 tasks rolled out in parallel; each rollout boots a sandbox (rLLM's
#     own SandboxedAgentFlow path, AgentFlowEngine) and runs terminus2
#     against it. The gateway routes every LLM call back to the trainer-hosted
#     model. This is NOT the remote-runtime / RemoteAgentFlowEngine path.
#   - 32K prompt window, 8K response budget per turn (terminus2 runs
#     many turns; the per-turn cap keeps the optimizer batch shape sane).
#
# Sandbox backend is chosen by SWE_SANDBOX_BACKEND (docker | local | modal |
# daytona; default modal). modal needs `pip install modal` + `modal token new`;
# daytona needs `pip install daytona` + DAYTONA_API_KEY. Per-rollout agent
# timeout: RLLM_HARNESS_RUN_TIMEOUT_S.
#
# Override anything by passing extra Hydra args after the script:
#   bash train_tinker.sh model.name=Qwen/Qwen3-8B training.group_size=4

set -euo pipefail

export SWE_SANDBOX_BACKEND="${SWE_SANDBOX_BACKEND:-modal}"
# Per-rollout turn cap for terminus2 (read by train.py). Empty = uncapped.
export TERMINUS_MAX_TURNS="${TERMINUS_MAX_TURNS:-100}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-1800}"

python -u train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3.5-4B \
    model.lora_rank=32 \
    training.group_size=8 \
    training.learning_rate=2e-5 \
    training.max_length=65536 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.rollout.val.temperature=1.0 \
    rllm.rollout.val.top_p=1.0 \
    data.max_prompt_length=57344 \
    data.max_response_length=8192 \
    data.train_batch_size=1 \
    data.val_batch_size=-1 \
    rllm.compact_filtering.enable=true \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.async_training.enable=true \
    rllm.async_training.mini_batch_size=16 \
    rllm.async_training.fwd_bwd_group_size=1 \
    rllm.async_training.staleness_threshold=0.5 \
    rllm.async_training.trigger_parameter_sync_step=1 \
    rllm.async_training.partial_rollout=true \
    rllm.workflow.n_parallel_tasks=128 \
    rllm.workflow.raise_on_error=false \
    rllm.gateway.port=9090 \
    rllm.gateway.cumulative_token_mode=true \
    rllm.renderer.family=qwen3.5 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='swe-rl' \
    rllm.trainer.experiment_name='r2egym-terminus2-qwen3.5-4b' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=10 \
    rllm.trainer.save_freq=10 \
    "$@"

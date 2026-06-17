#!/usr/bin/env bash
# Train an SWE agent on R2E-Gym, eval on SWE-bench Verified — Fireworks backend.
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
# What this configures, in plain English:
#   - Async GRPO with compact filtering (drop too-long rollouts before grad).
#   - 128 tasks rolled out in parallel; each rollout boots a sandbox (rLLM's
#     own SandboxedAgentFlow path, AgentFlowEngine) and runs terminus2
#     against it. The gateway routes every LLM call back to the Fireworks-hosted
#     deployment. This is NOT the remote-runtime / RemoteAgentFlowEngine path.
#   - 32K prompt window, 8K response budget per turn (terminus2 runs
#     many turns; the per-turn cap keeps the optimizer batch shape sane).
#
# Model: Qwen3.5-9B + LoRA-32 on the qwen3p5-9b-256k-lora training shape.
# Fireworks' public catalog ships a 3.5-9B LoRA shape but no 3.5-4B, so this
# uses the 9B the README names as the base model rather than the tinker recipe's
# 4B. To change it, swap model.name / model.tokenizer_model /
# fireworks_config.policy_trainer_shape_id together (see docs/backends/fireworks.mdx).
#
# Sandbox backend is chosen by SWE_SANDBOX_BACKEND (docker | local | modal |
# daytona; default modal). modal needs `pip install modal` + `modal token new`;
# daytona needs `pip install daytona` + DAYTONA_API_KEY. Per-rollout agent
# timeout: RLLM_HARNESS_RUN_TIMEOUT_S.
#
# Qwen3.5 is a reasoning family; if the harness can't parse reasoning output,
# disable it by appending: rollout_engine.reasoning_effort=none
#
# Override anything by passing extra Hydra args after the script:
#   bash train_fireworks.sh training.group_size=4

set -euo pipefail

export SWE_SANDBOX_BACKEND="${SWE_SANDBOX_BACKEND:-modal}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-1800}"

python -u train.py \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/qwen3p5-9b \
    model.tokenizer_model=Qwen/Qwen3.5-9B \
    model.lora_rank=32 \
    fireworks_config.policy_trainer_shape_id=accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora \
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
    rllm.gateway.renderer_family=qwen3.5 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='swe-rl' \
    rllm.trainer.experiment_name='r2egym-terminus2-qwen3p5-9b-fireworks' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=10 \
    rllm.trainer.save_freq=10 \
    "$@"

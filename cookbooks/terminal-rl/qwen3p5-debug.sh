#!/usr/bin/env bash
# qwen3p5-debug — Qwen3.5-35B-A3B LoRA debug run on the tb-v2-debug dataset (8 tasks).
#
# Fireworks backend + Modal sandboxes + terminus2. Snapshot of the tuned config:
#   - 32 rollouts per task (training.group_size) — GRPO/ECHO group of 32
#   - 8 task groups per optimizer step (rllm.async_training.mini_batch_size)
#     => 256 trajectories per step; tb-v2-debug (8 tasks) = exactly 1 step/epoch
#   - async in-flight cap: (1+staleness 3.0) x 8 = 32 groups (<=1024 rollouts),
#     throttled by the 256-sandbox cap (rllm.workflow.n_parallel_tasks)
#   - trainer + rollout deployment are auto-provisioned on Fireworks at startup
#     and DELETED at shutdown (also true for a reattached deployment_id)
#
# Before running:
#   export FIREWORKS_API_KEY=...     # training reads the env var
#   rllm dataset pull tb-v2-debug    # no-op once pulled
#   rllm tunnel up                   # Modal sandboxes reach the gateway (port 9090)
#
# Run (from anywhere; the script cd's itself):
#   bash cookbooks/terminal-rl/qwen3p5-debug.sh
# Override anything by appending Hydra args, e.g.:
#   bash qwen3p5-debug.sh fireworks_config.rollout_deployment_replica_count=4
# Reattach an existing rollout deployment (it is still deleted at shutdown):
#   bash qwen3p5-debug.sh fireworks_infra.deployments.rollout.deployment_id=accounts/rllm-project/deployments/<id>

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export TERMINAL_SANDBOX_BACKEND="${TERMINAL_SANDBOX_BACKEND:-modal}"
# Train dataset (DatasetRegistry name). Pull it first: rllm dataset pull <name>
export TB_TRAIN_DATASET="${TB_TRAIN_DATASET:-tb-v2-debug}"
# Per-rollout turn cap for terminus2 (read by train.py). Empty = uncapped.
export TERMINUS_MAX_TURNS="${TERMINUS_MAX_TURNS:-100}"
# Plain ReAct loop: no harbor summarization — a context overflow ends the episode.
export RLLM_TERMINUS_ENABLE_SUMMARIZE="${RLLM_TERMINUS_ENABLE_SUMMARIZE:-0}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-3600}"
# Modal sandbox LIFETIME (not idle time). Must exceed the agent run timeout
# above plus setup/verify, or sandboxes get reaped mid-rollout — surfacing as
# "Sandbox has already shut down" (NotFoundError) and exit-137 kills.
export RLLM_MODAL_SANDBOX_TIMEOUT_S="${RLLM_MODAL_SANDBOX_TIMEOUT_S:-4800}"

python -u train.py \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/qwen3p5-35b-a3b \
    model.tokenizer_model=Qwen/Qwen3.5-35B-A3B \
    model.lora_rank=32 \
    fireworks_config.policy_trainer_shape_id=accounts/fireworks/trainingShapes/qwen3p5-35b-a3b-256k-lora \
    fireworks_config.policy_trainer_replica_count=2 \
    fireworks_config.rollout_deployment_replica_count=6 \
    training.group_size=32 \
    training.learning_rate=2e-5 \
    training.max_length=101072 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.rollout.val.temperature=1.0 \
    rllm.rollout.val.top_p=1.0 \
    data.max_prompt_length=92880 \
    data.max_response_length=8192 \
    data.train_batch_size=1 \
    data.val_batch_size=-1 \
    rllm.data.max_prompt_length=92880 \
    rllm.data.max_response_length=8192 \
    rllm.data.train_batch_size=1 \
    rllm.data.val_batch_size=-1 \
    rllm.compact_filtering.enable=true \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.async_training.enable=true \
    rllm.async_training.mini_batch_size=8 \
    rllm.async_training.fwd_bwd_group_size=1 \
    rllm.async_training.staleness_threshold=3.0 \
    rllm.async_training.trigger_parameter_sync_step=1 \
    rllm.async_training.partial_rollout=true \
    rllm.workflow.n_parallel_tasks=256 \
    rllm.workflow.raise_on_error=false \
    rllm.rejection_sample.filter_uniform_groups=true \
    rllm.gateway.port=9091 \
    rllm.gateway.num_workers=4 \
    rllm.gateway.cumulative_token_mode=true \
    rllm.gateway.renderer_family=qwen3.5 \
    rllm.trainer.total_epochs=100 \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='terminal-rl' \
    rllm.trainer.experiment_name='qwen3p5-35b-a3b-tb-v2-debug' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=50 \
    rllm.trainer.save_freq=10 \
    "$@"

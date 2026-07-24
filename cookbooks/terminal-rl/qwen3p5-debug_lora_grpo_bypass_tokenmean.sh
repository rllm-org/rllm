#!/usr/bin/env bash
# qwen3p5-debug_lora_grpo_bypass_tokenmean — Qwen3.5-35B-A3B LoRA + native GRPO with
# rollout-logprob bypass and R3 router replay on the tb-v2-debug dataset (8 tasks).
#
# Fireworks backend + Modal sandboxes + native_react. Snapshot of the tuned config:
#   - 32 rollouts per task (training.group_size) — GRPO advantage group of 32
#   - Native PPO-clipped GRPO loss with epsilon=0.2; sampler logprobs are the old policy
#   - 8 task groups in one forward/backward pass per optimizer step
#     => 256 trajectories per step; tb-v2-debug (8 tasks) = exactly 1 step/epoch
#   - async in-flight cap: (1+staleness 3.0) x 8 = 32 groups (<=1024 rollouts),
#     throttled by the 256-sandbox cap (rllm.workflow.n_parallel_tasks)
#   - trainer + rollout deployment are auto-provisioned on Fireworks at startup
#     and DELETED at shutdown (also true for a reattached deployment_id)
#
# Before running:
#   export FIREWORKS_API_KEY=...     # training reads the env var
#   rllm dataset pull tb-v2-debug    # no-op once pulled
#   No tunnel is needed: native_react calls the gateway from the trainer host.
#
# Run (from anywhere; the script cd's itself):
#   bash cookbooks/terminal-rl/qwen3p5-debug_lora_grpo_bypass_tokenmean.sh
# Override anything by appending Hydra args, e.g.:
#   bash qwen3p5-debug_lora_grpo_bypass_tokenmean.sh fireworks_config.rollout_deployment_replica_count=4
# Reattach an existing rollout deployment (it is still deleted at shutdown):
#   bash qwen3p5-debug_lora_grpo_bypass_tokenmean.sh fireworks_infra.deployments.rollout.deployment_id=accounts/rllm-project/deployments/<id>

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Fireworks key: an exported FIREWORKS_API_KEY wins; otherwise fall back to the
# key on file in ~/.rllm/config.json (written by `rllm model setup`).
export FIREWORKS_API_KEY="${FIREWORKS_API_KEY:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.rllm/config.json')))['api_keys']['fireworks'])" 2>/dev/null || true)}"
if [ -z "${FIREWORKS_API_KEY}" ]; then
    echo "FIREWORKS_API_KEY is not set and ~/.rllm/config.json has no api_keys.fireworks (run: rllm model setup)" >&2
    exit 1
fi

export TERMINAL_SANDBOX_BACKEND="${TERMINAL_SANDBOX_BACKEND:-modal}"
export TERMINAL_AGENT="${TERMINAL_AGENT:-native_react}"
# Train dataset (DatasetRegistry name). Pull it first: rllm dataset pull <name>
export TB_TRAIN_DATASET="${TB_TRAIN_DATASET:-tb-v2-debug}"
# native_react is append-only: no compaction, and every assistant response
# field (including reasoning_content and tool_calls) is resent on later turns.
export NATIVE_REACT_MAX_TURNS="${NATIVE_REACT_MAX_TURNS:-100}"
# Cap sandbox resources below each task's declared ask (Modal bills reserved
# CPU+memory per second; a cap only LOWERS a task's declared value, never
# raises it). tb-v2 tasks declare up to 8 CPUs — capping at 0.125 cuts the CPU
# bill ~64x. Memory/storage caps can OOM compile-heavy graders; storage stays
# opt-in:
#   export RLLM_SANDBOX_MAX_STORAGE_MB=8192
export RLLM_SANDBOX_MAX_CPUS="${RLLM_SANDBOX_MAX_CPUS:-0.125}"
export RLLM_SANDBOX_MAX_MEMORY_MB="${RLLM_SANDBOX_MAX_MEMORY_MB:-256}"
# Host-side native_react does not expose the gateway to the sandbox.
export RLLM_TUNNEL_SPEC="${RLLM_TUNNEL_SPEC:-null}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-3600}"
# Sandbox LIFETIME floor, provider-agnostic (not idle time). Must exceed the agent run timeout
# above plus setup/verify, or sandboxes get reaped mid-rollout — surfacing as
# "Sandbox has already shut down" (NotFoundError) and exit-137 kills.
export RLLM_SANDBOX_TIMEOUT_S="${RLLM_SANDBOX_TIMEOUT_S:-4800}"

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
    training.beta2=0.999 \
    training.max_length=133072 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.rollout.val.temperature=1.0 \
    rllm.rollout.val.top_p=1.0 \
    data.max_prompt_length=100000 \
    data.max_response_length=32768 \
    data.train_batch_size=1 \
    data.val_batch_size=-1 \
    rllm.data.max_prompt_length=100000 \
    rllm.data.max_response_length=32768 \
    rllm.data.train_batch_size=1 \
    rllm.data.val_batch_size=-1 \
    rllm.compact_filtering.enable=true \
    rllm.compact_filtering.mask_max_prompt_length_exceeded=false \
    rllm.compact_filtering.mask_max_response_length_exceeded=false \
    rllm.compact_filtering.mask_max_turns_exceeded=false \
    rllm.compact_filtering.mask_timeout=false \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.router_replay=R3 \
    rllm.algorithm.loss_fn=grpo \
    rllm.algorithm.loss_agg_mode=token-mean \
    rllm.algorithm.eps_clip=0.2 \
    rllm.algorithm.rollout_correction.bypass_mode=true \
    rllm.algorithm.norm_adv_by_std_in_grpo=false \
    rllm.async_training.enable=true \
    rllm.async_training.mini_batch_size=8 \
    rllm.async_training.fwd_bwd_group_size=8 \
    rllm.async_training.staleness_threshold=3.0 \
    rllm.async_training.trigger_parameter_sync_step=1 \
    rllm.async_training.partial_rollout=true \
    rllm.workflow.n_parallel_tasks=256 \
    rllm.workflow.raise_on_error=false \
    rllm.rejection_sample.filter_uniform_groups=true \
    rllm.gateway.port=9091 \
    "rllm.gateway.tunnel=${RLLM_TUNNEL_SPEC}" \
    rllm.gateway.num_workers=8 \
    rllm.gateway.cumulative_token_mode=true \
    rllm.gateway.renderer_family=qwen3.5 \
    rllm.trainer.total_epochs=500 \
    rllm.trainer.dump_batch_dir=null \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='terminal-rl' \
    rllm.trainer.experiment_name='qwen3p5-35b-a3b-tb-v2-debug-lora-r3-grpo-bypass-tokenmean-b2-0999' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=-1 \
    rllm.trainer.save_freq=10 \
    "$@"

#!/usr/bin/env bash
# qwen3p5-debug_lora_dppo_bypass — Qwen3.5-35B-A3B LoRA + DPPO-TV with
# rollout-logprob bypass and R3 router replay on the tb-v2-debug dataset (8 tasks).
#
# Fireworks backend + Modal sandboxes + terminus2. Snapshot of the tuned config:
#   - 32 rollouts per task (training.group_size) — GRPO advantage group of 32
#   - DPPO-TV policy loss with delta=0.2; sampler logprobs are the old policy
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
#   rllm tunnel up                   # Modal sandboxes reach the gateway
#
# Run (from anywhere; the script cd's itself):
#   bash cookbooks/terminal-rl/qwen3p5-debug_lora_dppo_bypass.sh
# Override anything by appending Hydra args, e.g.:
#   bash qwen3p5-debug_lora_dppo_bypass.sh fireworks_config.rollout_deployment_replica_count=4
# Reattach an existing rollout deployment (it is still deleted at shutdown):
#   bash qwen3p5-debug_lora_dppo_bypass.sh fireworks_infra.deployments.rollout.deployment_id=accounts/rllm-project/deployments/<id>

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
# Train dataset (DatasetRegistry name). Pull it first: rllm dataset pull <name>
export TB_TRAIN_DATASET="${TB_TRAIN_DATASET:-tb-v2-debug}"
# Per-rollout turn cap for terminus2 (read by train.py). Empty = uncapped.
export TERMINUS_MAX_TURNS="${TERMINUS_MAX_TURNS:-100}"
# Plain ReAct loop: no harbor summarization — a context overflow ends the episode.
# (read by train.py and passed to the harness as enable_summarize)
export TERMINUS_ENABLE_SUMMARIZE="${TERMINUS_ENABLE_SUMMARIZE:-0}"
# Keep each turn's reasoning in chat history and resend it (interleaved
# thinking; read by train.py, passed to the harness). Set to 0 to strip.
export TERMINUS_INTERLEAVED_THINKING="${TERMINUS_INTERLEAVED_THINKING:-1}"
# Cap sandbox resources below each task's declared ask (Modal bills reserved
# CPU+memory per second; a cap only LOWERS a task's declared value, never
# raises it). tb-v2 tasks declare up to 8 CPUs — capping at 0.125 cuts the CPU
# bill ~64x. Memory/storage caps can OOM compile-heavy graders; storage stays
# opt-in:
#   export RLLM_SANDBOX_MAX_STORAGE_MB=8192
export RLLM_SANDBOX_MAX_CPUS="${RLLM_SANDBOX_MAX_CPUS:-0.125}"
export RLLM_SANDBOX_MAX_MEMORY_MB="${RLLM_SANDBOX_MAX_MEMORY_MB:-256}"
# Per-job tunnel endpoint: every concurrent job needs its OWN tunnel (reserved
# ngrok domain or "cloudflared"); sharing one endpoint misroutes sandbox traffic.
# The full-FT counterpart (qwen3p5-debug_fullft_r3.sh) uses thw2.ngrok.app.
export RLLM_TUNNEL_SPEC="${RLLM_TUNNEL_SPEC:-ngrok:thw1.ngrok.app}"
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
    fireworks_config.rollout_deployment_replica_count=8 \
    training.group_size=32 \
    training.learning_rate=2e-5 \
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
    rllm.algorithm.loss_fn=dppo_tv \
    '+rllm.algorithm.loss_params.delta=0.2' \
    rllm.algorithm.rollout_correction.bypass_mode=true \
    rllm.algorithm.norm_adv_by_std_in_grpo=false \
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
    "rllm.gateway.tunnel=${RLLM_TUNNEL_SPEC}" \
    rllm.gateway.num_workers=8 \
    rllm.gateway.cumulative_token_mode=true \
    rllm.gateway.renderer_family=qwen3.5 \
    rllm.trainer.total_epochs=500 \
    rllm.trainer.dump_batch_dir=null \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='terminal-rl' \
    rllm.trainer.experiment_name='qwen3p5-35b-a3b-tb-v2-debug-lora-r3-dppo-tv-bypass' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=-1 \
    rllm.trainer.save_freq=10 \
    "$@"

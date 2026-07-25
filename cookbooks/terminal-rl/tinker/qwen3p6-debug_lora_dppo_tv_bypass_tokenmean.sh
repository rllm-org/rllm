#!/usr/bin/env bash
# qwen3p6-debug_lora_dppo_tv_bypass_tokenmean — Qwen3.6-35B-A3B LoRA with GRPO
# advantages, DPPO-TV loss, and rollout-logprob bypass.
#
# Tinker backend + Modal sandboxes + native_react. This is the Fireworks launcher
# copied verbatim and then incrementally adapted for Tinker:
#   - 32 rollouts per task (training.group_size) — GRPO advantage group of 32
#   - DPPO-TV policy loss with delta=0.2; sampler logprobs are the old policy
#   - 8 task groups in one forward/backward pass per optimizer step
#     => 256 trajectories per step; tb-v2-debug (8 tasks) = exactly 1 step/epoch
#   - async in-flight cap: (1+staleness 3.0) x 8 = 32 groups (<=1024 rollouts),
#     throttled by the 256-sandbox cap (rllm.workflow.n_parallel_tasks)
#   - 64K total context; response generation is capped at 32K and dynamically
#     clipped to the context remaining after the cumulative prompt
#   - Tinker provisions the trainer and sampler; there are no shape or replica IDs
#
# Before running:
#   export TINKER_API_KEY=...        # training reads the env var
#   export TINKER_PROJECT_ID=...     # optional project scope
#   rllm dataset pull tb-v2-debug    # no-op once pulled
#   native_react itself needs no tunnel; the existing per-job tunnel default is
#   retained below for compatibility with the current launcher workflow.
#
# Run (from anywhere; the script cd's itself):
#   bash cookbooks/terminal-rl/tinker/qwen3p6-debug_lora_dppo_tv_bypass_tokenmean.sh
# Override anything by appending Hydra args, e.g.:
#   bash cookbooks/terminal-rl/tinker/qwen3p6-debug_lora_dppo_tv_bypass_tokenmean.sh training.group_size=16

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# An exported TINKER_API_KEY wins; otherwise use the key stored by rLLM.
export TINKER_API_KEY="${TINKER_API_KEY:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.rllm/config.json')))['api_keys']['tinker'])" 2>/dev/null || true)}"
if [ -z "${TINKER_API_KEY}" ]; then
    echo "TINKER_API_KEY is not set and ~/.rllm/config.json has no api_keys.tinker" >&2
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
# Modal's create quota is account-wide, while rLLM's limiter is process-local.
# Keep this job below the shared 5/s cap when other training jobs are active.
export RLLM_MODAL_SANDBOX_CREATE_RPS="${RLLM_MODAL_SANDBOX_CREATE_RPS:-1}"
export RLLM_MODAL_SANDBOX_CREATE_BURST="${RLLM_MODAL_SANDBOX_CREATE_BURST:-8}"
# Per-job tunnel endpoint: every concurrent job needs its own reserved
# ngrok domain (or cloudflared endpoint); sharing one misroutes gateway traffic.
export RLLM_TUNNEL_SPEC="${RLLM_TUNNEL_SPEC:-ngrok:thw2.ngrok.app}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-3600}"
# Sandbox LIFETIME floor, provider-agnostic (not idle time). Must exceed the agent run timeout
# above plus setup/verify, or sandboxes get reaped mid-rollout — surfacing as
# "Sandbox has already shut down" (NotFoundError) and exit-137 kills.
export RLLM_SANDBOX_TIMEOUT_S="${RLLM_SANDBOX_TIMEOUT_S:-4800}"

python -u train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3.6-35B-A3B \
    model.tokenizer_model=Qwen/Qwen3.6-35B-A3B \
    model.lora_rank=32 \
    model.train_unembed=false \
    model.train_attn=true \
    model.train_mlp=true \
    fuse_forward_backward_and_optim_step=false \
    training.group_size=32 \
    training.learning_rate=2e-5 \
    training.beta1=0.9 \
    training.beta2=0.999 \
    training.eps=1e-8 \
    training.weight_decay=0.01 \
    training.grad_clip_norm=1.0 \
    training.max_length=65536 \
    training.num_minibatches=1 \
    training.resume_mode=auto \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.rollout.val.temperature=1.0 \
    rllm.rollout.val.top_p=1.0 \
    data.max_prompt_length=57344 \
    data.max_response_length=32768 \
    data.train_batch_size=1 \
    data.val_batch_size=-1 \
    rllm.data.max_prompt_length=57344 \
    rllm.data.max_response_length=32768 \
    rllm.data.train_batch_size=1 \
    rllm.data.val_batch_size=-1 \
    rllm.compact_filtering.enable=true \
    rllm.compact_filtering.mask_max_prompt_length_exceeded=false \
    rllm.compact_filtering.mask_max_response_length_exceeded=false \
    rllm.compact_filtering.mask_max_turns_exceeded=false \
    rllm.compact_filtering.mask_timeout=false \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.router_replay=disabled \
    rllm.algorithm.loss_fn=dppo_tv \
    rllm.algorithm.loss_agg_mode=token-mean \
    '+rllm.algorithm.loss_params.delta=0.2' \
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
    rllm.gateway.num_workers=0 \
    rllm.gateway.cumulative_token_mode=true \
    rllm.gateway.renderer_family=qwen3.6 \
    rllm.trainer.total_epochs=500 \
    rllm.trainer.dump_batch_dir=null \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='terminal-rl' \
    rllm.trainer.experiment_name='qwen3p6-35b-a3b-tb-v2-debug-tinker-lora-dppo-tv-bypass-tokenmean-b2-0999-64k' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=-1 \
    rllm.trainer.save_freq=10 \
    "$@"

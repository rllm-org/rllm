#!/usr/bin/env bash
# Reproduce https://wandb.ai/agentica/terminal-rl/runs/z6xi3xkh on the
# current terminal-rl branch, with raw TraceRecord -> TraceGraph parity dumps.
#
# Source of truth:
#   - W&B recorded program/config from commit d501e357967c45bb46807d5f6a3366cbcdfee32b
#   - W&B run arguments are copied below field-for-field
#   - W&B captured transformers==5.15.0; pin it because this branch's 5.5.3
#     does not recognize model_type=deepseek_v4
#
# Intentional current-branch adaptations:
#   - tb_v2_debug -> the current registry name tb-v2-debug
#   - the old machine's fixed http://5.78.144.17:19090 route is replaced by
#     current automatic tunnel resolution (a registered ngrok wildcard creates
#     a unique hostname and gateway port for every run)
#   - gateway store=compact plus a required, unique parity dump directory
#   - nested Hydra syntax for loss_params, whose parent now exists by default
#   - a shorter experiment name avoids the original run's >63-character
#     Fireworks output-model-id checkpoint promotion failure
#
# This script deliberately keeps MiniSweAgentHarness, cumulative-token mode,
# group/async sizing, model, optimizer, and rollout settings from the W&B run.
# It runs the parity verifier after training exits, including failed runs.
#
# Required before launch:
#   export FIREWORKS_API_KEY=...
#   rllm tunnel setup  # once; register an ngrok wildcard such as *.example.com
#   rllm dataset pull tb-v2-debug
#   rllm dataset pull harbor:terminal-bench@2.0
#
# Raw records and graphs default to:
#   /data/home/thw/trace-dumps/z6xi3xkh-parity-002
# Set RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR only to override that path.
#
# With a registered wildcard, do not run `rllm tunnel up` and do not pass a
# tunnel or port here. AgentTrainer creates both automatically. To deliberately
# override that behavior, set the standard RLLM_GATEWAY_TUNNEL environment
# variable before launch.
#
# Review without launching anything:
#   RLLM_LAUNCH_DRY_RUN=1 \
#     RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR=/tmp/z6xi3xkh-review \
#     bash cookbooks/terminal-rl/reproduce_z6xi3xkh.sh
# Set RLLM_LAUNCH_CONFIG_ONLY=1 instead to make Hydra parse and print the fully
# resolved static job config without entering train_debug.main or provisioning
# infra. Tunnel and port remain null in that output because AgentTrainer resolves
# them dynamically after loading the datasets.
#
# A bounded first launch can append a normal Hydra override, for example:
#   ... bash cookbooks/terminal-rl/reproduce_z6xi3xkh.sh rllm.trainer.total_batches=1

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

export RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR="${RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR:-/data/home/thw/trace-dumps/z6xi3xkh-parity-002}"

if [[ "$RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR" != /* ]]; then
    echo "RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR must be an absolute path" >&2
    exit 2
fi

export TERMINAL_SANDBOX_BACKEND="${TERMINAL_SANDBOX_BACKEND:-modal}"
export TB_TRAIN_DATASET="${TB_TRAIN_DATASET:-tb-v2-debug}"
export MINISWE_MAX_TURNS="${MINISWE_MAX_TURNS:-64}"
export MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS="${MINISWE_MAX_CONSECUTIVE_FORMAT_ERRORS:-1}"
export MINISWE_COMMAND_TIMEOUT="${MINISWE_COMMAND_TIMEOUT:-300}"
export RLLM_SANDBOX_MAX_CPUS="${RLLM_SANDBOX_MAX_CPUS:-0.125}"
export RLLM_SANDBOX_MAX_MEMORY_MB="${RLLM_SANDBOX_MAX_MEMORY_MB:-256}"
export RLLM_MODAL_SANDBOX_CREATE_RPS="${RLLM_MODAL_SANDBOX_CREATE_RPS:-2}"
export RLLM_HARNESS_INSTALL_TIMEOUT_S="${RLLM_HARNESS_INSTALL_TIMEOUT_S:-300}"
export RLLM_HARNESS_RUN_TIMEOUT_S="${RLLM_HARNESS_RUN_TIMEOUT_S:-2400}"
export RLLM_HARNESS_VERIFIER_TIMEOUT_S="${RLLM_HARNESS_VERIFIER_TIMEOUT_S:-300}"
export RLLM_SANDBOX_TIMEOUT_S="${RLLM_SANDBOX_TIMEOUT_S:-3000}"
export RLLM_EXPERIMENT_NAME="${RLLM_EXPERIMENT_NAME:-z6xi3xkh-dsv4-flash-lora128-trace-parity}"

launch_cmd=(
    uv run --with transformers==5.15.0 python -u cookbooks/terminal-rl/train_debug.py
    rllm/backend=fireworks
    model.name=accounts/fireworks/models/deepseek-v4-flash-0731
    model.tokenizer_model=deepseek-ai/DeepSeek-V4-Flash-0731
    model.lora_rank=128
    fireworks_config.policy_trainer_shape_id=accounts/fireworks/trainingShapes/deepseek-v4-flash-0731-256k-lora
    fireworks_config.policy_trainer_replica_count=2
    fireworks_config.rollout_deployment_replica_count=4
    training.group_size=16
    training.learning_rate=5e-5
    training.grad_clip_norm=0.0
    training.beta2=0.999
    training.eps=1e-10
    training.max_length=67584
    rllm.rollout.train.temperature=1.0
    rllm.rollout.train.top_p=1.0
    rllm.rollout.val.temperature=1.0
    rllm.rollout.val.top_p=1.0
    data.max_prompt_length=67584
    data.max_response_length=16384
    data.train_batch_size=1
    data.val_batch_size=-1
    rllm.data.max_prompt_length=67584
    rllm.data.max_response_length=16384
    rllm.data.train_batch_size=1
    rllm.data.val_batch_size=-1
    rllm.compact_filtering.enable=false
    rllm.algorithm.adv_estimator=grpo
    rllm.algorithm.norm_adv_by_std_in_grpo=false
    rllm.algorithm.router_replay=R3
    rllm.algorithm.loss_fn=dppo_tv
    +rllm.algorithm.loss_params.delta=0.1
    rllm.algorithm.loss_agg_mode=token-mean
    rllm.algorithm.rollout_correction.bypass_mode=true
    rllm.async_training.enable=true
    rllm.async_training.mini_batch_size=8
    rllm.async_training.fwd_bwd_group_size=8
    rllm.async_training.staleness_threshold=3.0
    rllm.async_training.trigger_parameter_sync_step=1
    rllm.async_training.partial_rollout=true
    rllm.workflow.n_parallel_tasks=384
    rllm.workflow.raise_on_error=false
    rllm.workflow.verify_only_on_env_done=true
    rllm.rejection_sample.filter_uniform_groups=true
    rllm.rejection_sample.refill_filtered_uniform_groups=true
    rllm.gateway.host=127.0.0.1
    rllm.gateway.num_workers=4
    rllm.gateway.store=compact
    "rllm.gateway.trace_parity_dump_dir=${RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR}"
    rllm.gateway.cumulative_token_mode=true
    rllm.gateway.renderer_family=auto
    rllm.trainer.total_epochs=1000
    rllm.episode_logging.log_episodes=true
    "rllm.trainer.logger=[wandb]"
    rllm.trainer.project_name=terminal-rl
    "rllm.trainer.experiment_name=${RLLM_EXPERIMENT_NAME}"
    rllm.trainer.val_before_train=false
    rllm.trainer.test_freq=-1
    rllm.trainer.save_freq=20
    "$@"
)

if [[ "${RLLM_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
    printf 'Launch command:'
    printf ' %q' "${launch_cmd[@]}"
    printf '\nParity verifier: uv run python scripts/verify_trace_parity_dump.py %q\n' "$RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR"
    exit 0
fi

if [[ "${RLLM_LAUNCH_CONFIG_ONLY:-0}" == "1" ]]; then
    "${launch_cmd[@]}" --cfg job --resolve
    exit 0
fi

export FIREWORKS_API_KEY="${FIREWORKS_API_KEY:-$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.rllm/config.json')))['api_keys']['fireworks'])" 2>/dev/null || true)}"
if [[ -z "$FIREWORKS_API_KEY" ]]; then
    echo "FIREWORKS_API_KEY is not set and ~/.rllm/config.json has no api_keys.fireworks (run: rllm model setup)" >&2
    exit 2
fi

if [[ -d "$RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR" ]] && find "$RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "Trace parity dump directory must be empty: $RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR" >&2
    exit 2
fi
mkdir -p "$RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR"

printf 'Raw TraceRecord and TraceGraph dump: %s\n' "$RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR"
printf 'Tunnel: automatic (registered ngrok wildcard; RLLM_GATEWAY_TUNNEL may override)\n'

training_status=0
"${launch_cmd[@]}" || training_status=$?

verify_status=0
uv run python scripts/verify_trace_parity_dump.py "$RLLM_GATEWAY_TRACE_PARITY_DUMP_DIR" || verify_status=$?

if ((training_status != 0)); then
    exit "$training_status"
fi
exit "$verify_status"

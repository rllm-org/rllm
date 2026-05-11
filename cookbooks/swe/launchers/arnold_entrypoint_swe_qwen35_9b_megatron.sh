#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/opt/tiger/modelchef}
RLLM_ROOT="$REPO_ROOT/submodules/rllm"
COOKBOOK_DIR="$RLLM_ROOT/cookbooks/swe"
SCRIPT_DIR="$COOKBOOK_DIR/launchers"

cd "$REPO_ROOT"

# shellcheck source=swe_artifact_utils.sh
source "$SCRIPT_DIR/swe_artifact_utils.sh"

echo "============================================================"
echo "rLLM SWE Qwen3.5-9B Megatron Arnold entrypoint"
echo "host=$(hostname) role=${ARNOLD_ROLE:-unknown} task=${ARNOLD_ID:-unknown}"
echo "artifact_dir=$(artifact_dir)"
echo "mode=${RLLM_SWE_MODE:-smoke}"
echo "============================================================"

if [ "${RLLM_SWE_MODE:-smoke}" = "cluster_only" ]; then
    echo "Cluster-only mode: Arnold is holding the Ray cluster for an external SWE driver."
    echo "External driver should connect through the Ray Client port exposed by the current head pod."
    while true; do
        sleep 3600
    done
fi

restore_swe_artifacts
if [ "${RLLM_SWE_LINK_LOCAL_MODEL_PATH:-1}" = "1" ] \
    && [ ! -d /mnt/hdfs/model_path ] \
    && [ -d /tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots ]; then
    MODEL_SNAPSHOT="$(find -L /tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots -mindepth 1 -maxdepth 1 -type d -print -quit)"
    if [ -n "$MODEL_SNAPSHOT" ] && [ -d "$MODEL_SNAPSHOT" ]; then
        sudo mkdir -p /mnt/hdfs 2>/dev/null || mkdir -p /mnt/hdfs
        sudo ln -sfn "$MODEL_SNAPSHOT" /mnt/hdfs/model_path 2>/dev/null || ln -sfn "$MODEL_SNAPSHOT" /mnt/hdfs/model_path
    fi
fi
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$COOKBOOK_DIR:$RLLM_ROOT:${PYTHONPATH:-}"
export MODEL_PATH=${MODEL_PATH:-/mnt/hdfs/model_path}
export RLLM_SWE_OUTPUT_DIR=${RLLM_SWE_OUTPUT_DIR:-/tmp/rllm_swe_outputs}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$RLLM_SWE_OUTPUT_DIR/checkpoints}
export TRAJ_DIR=${TRAJ_DIR:-$RLLM_SWE_OUTPUT_DIR/trajectories/\${trainer.experiment_name}}
export RLLM_RUN_TASK_RUNNER_LOCAL=1
export RAY_ADDRESS=${RAY_ADDRESS:-auto}
export LOGGER=${LOGGER:-"[console,wandb]"}
export SWE_REX_MODAL_APP_LOOKUP_TIMEOUT_S="${SWE_REX_MODAL_APP_LOOKUP_TIMEOUT_S:-60}"
export SWE_REX_MODAL_SANDBOX_CREATE_TIMEOUT_S="${SWE_REX_MODAL_SANDBOX_CREATE_TIMEOUT_S:-240}"
export SWE_REX_MODAL_TUNNELS_TIMEOUT_S="${SWE_REX_MODAL_TUNNELS_TIMEOUT_S:-60}"

if [ "${RLLM_SWE_MODE:-smoke}" = "external_ray_worker" ]; then
    if [ -z "${EXTERNAL_RAY_HEAD_IP:-}" ]; then
        echo "Missing EXTERNAL_RAY_HEAD_IP for external_ray_worker mode." >&2
        exit 2
    fi
    nvidia-smi
    "$VIRTUAL_ENV/bin/ray" stop --force || true
    sleep 2
    "$VIRTUAL_ENV/bin/ray" start \
        --address="[${EXTERNAL_RAY_HEAD_IP}]:${EXTERNAL_RAY_HEAD_PORT:-6379}" \
        --num-cpus="${RAY_WORKER_CPUS:-240}" \
        --num-gpus="${RAY_WORKER_GPUS:-8}" \
        --disable-usage-stats
    "$VIRTUAL_ENV/bin/ray" status || true
    echo "EXTERNAL_RAY_WORKER_READY host=$(hostname) role=${ARNOLD_ROLE:-unknown} head=${EXTERNAL_RAY_HEAD_IP}"
    while true; do
        "$VIRTUAL_ENV/bin/ray" status || true
        sleep 300
    done
fi

if [ -z "${MODAL_TOKEN_ID:-}" ] || [ -z "${MODAL_TOKEN_SECRET:-}" ]; then
    echo "Missing MODAL_TOKEN_ID or MODAL_TOKEN_SECRET." >&2
    exit 2
fi
if [ -z "${WANDB_API_KEY:-}" ] && [ "${RLLM_SWE_REQUIRE_WANDB:-1}" = "1" ]; then
    echo "Missing WANDB_API_KEY." >&2
    exit 2
fi

echo "Static network checks"
hostname || true
ip -br addr || true
ip route || true
ip -6 route || true
cat /etc/resolv.conf || true
getent ahosts api.modal.com | head || true
curl -sS -o /dev/null -w 'remote=%{remote_ip} connect=%{time_connect} total=%{time_total} code=%{http_code}\n' https://api.modal.com || true

cd "$COOKBOOK_DIR"
PROBE_OUT="/tmp/modal_probe_${ARNOLD_TRIAL_ID:-unknown}_${ARNOLD_ID:-0}.jsonl"
SWE_REX_REMOTE_RETRIES=0 python -m swe.scripts.modal_swerex_reliability_test \
    --total "${RLLM_SWE_MODAL_PROBE_TOTAL:-12}" \
    --concurrency "${RLLM_SWE_MODAL_PROBE_CONCURRENCY:-6}" \
    --mode light \
    --out "$PROBE_OUT"

python - "$PROBE_OUT" <<'PY'
import json
import sys

path = sys.argv[1]
summary = None
with open(path, encoding="utf-8") as handle:
    for line in handle:
        rec = json.loads(line)
        if rec.get("event") == "summary":
            summary = rec
if summary is None:
    raise SystemExit("BAD_CPU_DRIVER no summary")
p95 = summary.get("create_env_s", {}).get("p95")
p50 = summary.get("create_env_s", {}).get("p50")
ok = (
    summary.get("ok") == summary.get("total") == 12
    and summary.get("failed") == 0
    and summary.get("transient_failed") == 0
    and p95 is not None
    and p95 <= 30
    and (p50 is None or p50 <= 60)
)
print("modal_probe_summary=" + json.dumps(summary, sort_keys=True), flush=True)
if not ok:
    raise SystemExit("BAD_CPU_DRIVER")
PY

TRAINING_SCRIPT="$COOKBOOK_DIR/swe/training_scripts/run_swe_training_9b_megatron.sh"
if [ "${RLLM_SWE_MODE:-smoke}" = "smoke" ]; then
    export LOGGER=${LOGGER:-"[console]"}
    export ACTOR_LR_WARMUP_STEPS="${ACTOR_LR_WARMUP_STEPS:-0}"
    export SWE_STEP_LIMIT="${SWE_STEP_LIMIT:-24}"
    export SWE_AGENT_TIMEOUT="${SWE_AGENT_TIMEOUT:-180}"
    export SWE_COMMAND_TIMEOUT="${SWE_COMMAND_TIMEOUT:-60}"
    export SWE_SANDBOX_TIMEOUT="${SWE_SANDBOX_TIMEOUT:-240}"
    export SWE_STARTUP_JITTER_S="${SWE_STARTUP_JITTER_S:-0}"
    export SWE_VAL_STEP_LIMIT="${SWE_VAL_STEP_LIMIT:-24}"
    export SWE_VAL_AGENT_TIMEOUT="${SWE_VAL_AGENT_TIMEOUT:-180}"
    export SWE_VAL_COMMAND_TIMEOUT="${SWE_VAL_COMMAND_TIMEOUT:-60}"
    export SWE_VAL_SANDBOX_TIMEOUT="${SWE_VAL_SANDBOX_TIMEOUT:-240}"
    export SWE_VAL_STARTUP_JITTER_S="${SWE_VAL_STARTUP_JITTER_S:-0}"
    exec bash "$TRAINING_SCRIPT" \
        train_max_samples="${RLLM_SWE_SMOKE_TRAIN_SAMPLES:-16}" \
        val_max_samples="${RLLM_SWE_SMOKE_VAL_SAMPLES:-4}" \
        actor_rollout_ref.rollout.n="${RLLM_SWE_SMOKE_ROLLOUT_N:-1}" \
        rllm.workflow.n_parallel_tasks="${RLLM_SWE_SMOKE_PARALLEL_TASKS:-4}" \
        ++trainer.total_training_steps="${RLLM_SWE_SMOKE_STEPS:-1}" \
        trainer.total_epochs=1 \
        trainer.save_freq=1000 \
        trainer.test_freq=1000
fi

exec bash "$TRAINING_SCRIPT"

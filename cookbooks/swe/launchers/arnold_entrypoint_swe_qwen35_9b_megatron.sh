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
    setup_b200_driver_libs
    restore_verl_venv
    restore_rllm_home
else
    restore_swe_artifacts
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

if [ "${RLLM_SWE_MODE:-smoke}" = "cluster_only" ]; then
    echo "Cluster-only mode: Arnold is holding the Ray cluster for an external SWE driver."
    echo "External driver should connect with RAY_ADDRESS=ray://[<head_ipv6>]:10001."
    while true; do
        sleep 3600
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

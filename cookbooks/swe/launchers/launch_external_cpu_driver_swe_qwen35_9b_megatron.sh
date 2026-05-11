#!/usr/bin/env bash
# Launch the SWE Qwen3.5-9B Megatron driver on a Modal-good CPU Ray head.
#
# Expected topology:
#   1. This script runs on the direct CPU worker that owns the Ray head.
#   2. Arnold B200 workers have already joined that Ray head with
#      RLLM_SWE_MODE=external_ray_worker.
#   3. Run with ARNOLD_RAY_ADDRESS=auto when the driver is on the Ray head.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RLLM_ROOT="$(cd "$COOKBOOK_DIR/../.." && pwd)"
if [ -z "${REPO_ROOT:-}" ]; then
    MODELCHEF_ROOT_CANDIDATE="$(cd "$RLLM_ROOT/../.." 2>/dev/null && pwd)"
    if [ -d "$MODELCHEF_ROOT_CANDIDATE/submodules/rllm" ]; then
        REPO_ROOT="$MODELCHEF_ROOT_CANDIDATE"
    else
        REPO_ROOT="$RLLM_ROOT"
    fi
fi

cd "$REPO_ROOT"

# shellcheck source=swe_artifact_utils.sh
source "$SCRIPT_DIR/swe_artifact_utils.sh"

restore_verl_venv
restore_qwen35_cache
restore_rllm_home

MODEL_SNAPSHOT="${MODEL_PATH:-}"
if [ -z "$MODEL_SNAPSHOT" ] && [ -d /tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots ]; then
    MODEL_SNAPSHOT="$(find -L /tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots -mindepth 1 -maxdepth 1 -type d -print -quit)"
fi
if [ -z "$MODEL_SNAPSHOT" ] || [ ! -d "$MODEL_SNAPSHOT" ]; then
    echo "Missing Qwen3.5-9B model snapshot. Set MODEL_PATH or restore the HF cache artifact." >&2
    exit 2
fi

if [ ! -d /mnt/hdfs/model_path ]; then
    sudo mkdir -p /mnt/hdfs 2>/dev/null || mkdir -p /mnt/hdfs
    sudo ln -sfn "$MODEL_SNAPSHOT" /mnt/hdfs/model_path 2>/dev/null || ln -sfn "$MODEL_SNAPSHOT" /mnt/hdfs/model_path
fi
export MODEL_PATH=/mnt/hdfs/model_path

set +x
eval "$(
    python3 - <<'PY'
import netrc
import os
import shlex
import tomllib
from pathlib import Path

exports = {}
modal_path = Path("~/.modal.toml").expanduser()
if not os.environ.get("MODAL_TOKEN_ID") or not os.environ.get("MODAL_TOKEN_SECRET"):
    if not modal_path.exists():
        raise SystemExit("Missing ~/.modal.toml and Modal env vars are not set")
    modal = tomllib.loads(modal_path.read_text())
    active_profile = None
    for values in modal.values():
        if isinstance(values, dict) and values.get("active"):
            active_profile = values
            break
    if active_profile is None and len(modal) == 1:
        active_profile = next(iter(modal.values()))
    if not isinstance(active_profile, dict):
        raise SystemExit("Could not find an active Modal profile in ~/.modal.toml")
    exports["MODAL_TOKEN_ID"] = os.environ.get("MODAL_TOKEN_ID") or active_profile.get("token_id")
    exports["MODAL_TOKEN_SECRET"] = os.environ.get("MODAL_TOKEN_SECRET") or active_profile.get("token_secret")

if not os.environ.get("WANDB_API_KEY"):
    try:
        auth = netrc.netrc().authenticators("api.wandb.ai")
    except FileNotFoundError:
        auth = None
    if auth is not None:
        exports["WANDB_API_KEY"] = auth[2]

for key, value in exports.items():
    if not value:
        raise SystemExit(f"Credential source did not provide {key}")
    print(f"export {key}={shlex.quote(value)}")
PY
)"

if [ -z "${MODAL_TOKEN_ID:-}" ] || [ -z "${MODAL_TOKEN_SECRET:-}" ]; then
    echo "Missing MODAL_TOKEN_ID or MODAL_TOKEN_SECRET." >&2
    exit 2
fi
if [ -z "${WANDB_API_KEY:-}" ] && [ "${RLLM_SWE_REQUIRE_WANDB:-0}" = "1" ]; then
    echo "Missing WANDB_API_KEY." >&2
    exit 2
fi
set -x

export RAY_ADDRESS="${ARNOLD_RAY_ADDRESS:-${RAY_ADDRESS:-auto}}"
export RLLM_RUN_TASK_RUNNER_LOCAL=1
export RLLM_RAY_WORKING_DIR="${RLLM_RAY_WORKING_DIR:-$RLLM_ROOT}"
export NNODES="${NNODES:-2}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
export ACTOR_TP="${ACTOR_TP:-2}"
export ACTOR_CP="${ACTOR_CP:-2}"
export ACTOR_PP="${ACTOR_PP:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export MODEL_USE_REMOVE_PADDING="${MODEL_USE_REMOVE_PADDING:-true}"
export RLLM_VLLM_PORT_BASE="${RLLM_VLLM_PORT_BASE:-46000}"
export RLLM_VLLM_PORT_STRIDE="${RLLM_VLLM_PORT_STRIDE:-100}"
export LOGGER="${LOGGER:-[console]}"
export RLLM_SWE_OUTPUT_DIR="${RLLM_SWE_OUTPUT_DIR:-/tmp/rllm_swe_outputs_external}"
if [ "${RLLM_SWE_DRIVER_MODE:-smoke}" = "smoke" ]; then
    export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RLLM_SWE_OUTPUT_DIR/checkpoints}"
else
    export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/hdfs/swe_checkpoints}"
    if [ -n "${CHECKPOINT_HDFS_URI:-}" ] && ! mountpoint -q "$CHECKPOINT_ROOT"; then
        sudo mkdir -p "$CHECKPOINT_ROOT" 2>/dev/null || mkdir -p "$CHECKPOINT_ROOT"
        mlx hdfs mount --access-mode=RW "$CHECKPOINT_HDFS_URI" "$CHECKPOINT_ROOT"
    fi
    case "$CHECKPOINT_ROOT" in
        /mnt/hdfs/*) ;;
        *)
            echo "Full driver mode requires CHECKPOINT_ROOT to be a writable HDFS mount under /mnt/hdfs." >&2
            echo "Set CHECKPOINT_ROOT=/mnt/hdfs/<mount> and mount it with mlx hdfs, or set CHECKPOINT_HDFS_URI for this launcher." >&2
            exit 2
            ;;
    esac
    if ! mountpoint -q "$CHECKPOINT_ROOT"; then
        echo "CHECKPOINT_ROOT is not an HDFS/FUSE mountpoint: $CHECKPOINT_ROOT" >&2
        echo "For full mode, mount a writable HDFS path, for example:" >&2
        echo "  mlx hdfs mount --access-mode=RW hdfs://harunava/home/<user-or-team>/rllm_swe_checkpoints $CHECKPOINT_ROOT" >&2
        exit 2
    fi
    if [ ! -w "$CHECKPOINT_ROOT" ]; then
        echo "CHECKPOINT_ROOT mount is not writable: $CHECKPOINT_ROOT" >&2
        echo "For full mode, mount a writable HDFS path, for example:" >&2
        echo "  mlx hdfs mount --access-mode=RW hdfs://harunava/home/<user-or-team>/rllm_swe_checkpoints $CHECKPOINT_ROOT" >&2
        exit 2
    fi
fi
export TRAJ_DIR="${TRAJ_DIR:-$RLLM_SWE_OUTPUT_DIR/trajectories/\${trainer.experiment_name}}"

# Bound Modal control-plane calls so transient App/Sandbox stalls raise and
# flow through the existing sandbox retry loop.
export SWE_REX_MODAL_APP_LOOKUP_TIMEOUT_S="${SWE_REX_MODAL_APP_LOOKUP_TIMEOUT_S:-60}"
export SWE_REX_MODAL_SANDBOX_CREATE_TIMEOUT_S="${SWE_REX_MODAL_SANDBOX_CREATE_TIMEOUT_S:-240}"
export SWE_REX_MODAL_TUNNELS_TIMEOUT_S="${SWE_REX_MODAL_TUNNELS_TIMEOUT_S:-60}"

TRAINING_SCRIPT="$COOKBOOK_DIR/swe/training_scripts/run_swe_training_9b_megatron.sh"

if [ "${RLLM_SWE_DRIVER_MODE:-smoke}" = "smoke" ]; then
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

    cd "$COOKBOOK_DIR"
    exec bash "$TRAINING_SCRIPT" \
        train_max_samples="${RLLM_SWE_SMOKE_TRAIN_SAMPLES:-16}" \
        val_max_samples="${RLLM_SWE_SMOKE_VAL_SAMPLES:-4}" \
        actor_rollout_ref.rollout.n="${RLLM_SWE_SMOKE_ROLLOUT_N:-1}" \
        rllm.workflow.n_parallel_tasks="${RLLM_SWE_SMOKE_PARALLEL_TASKS:-4}" \
        ++trainer.total_training_steps="${RLLM_SWE_SMOKE_STEPS:-1}" \
        trainer.total_epochs=1 \
        trainer.save_freq=1000 \
        trainer.test_freq=1000 \
        trainer.val_before_train=false \
        ++rllm.trainer.val_before_train=false \
        "$@"
fi

cd "$COOKBOOK_DIR"
exec bash "$TRAINING_SCRIPT" "$@"

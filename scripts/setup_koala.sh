#!/bin/bash
# ============================================================================
# rLLM Koala Setup — shared orchestrator with pluggable env configs
# ============================================================================
# Usage:
#   . scripts/setup_koala.sh --env math              # restore from S3 cache
#   . scripts/setup_koala.sh --env math --fast       # skip already-present resources
#   . scripts/setup_koala.sh --upload-cache          # upload verified env to S3
#
# Flow:
#   Phase 1 (debug pod): manual uv sync + run training to validate
#   Phase 2 (same pod):  . scripts/setup_koala.sh --upload-cache
#   Phase 3 (normal pod): . scripts/setup_koala.sh --env math
#
# Env vars (set before sourcing or export in submit command):
#   HF_TOKEN       HuggingFace auth (gated datasets)
#   WANDB_API_KEY  WandB auth (optional)
#   EXP_NAME       Experiment name (required for S3 background sync)
# ============================================================================
set -euo pipefail

# --- Arg parsing ---
ENV_NAME="math"
FAST_MODE=0
UPLOAD_CACHE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)          ENV_NAME="$2"; shift 2 ;;
        --fast)         FAST_MODE=1; shift ;;
        --upload-cache) UPLOAD_CACHE=1; shift ;;
        *)              echo "ERROR: Unknown option: $1"; return 1 2>/dev/null || exit 1 ;;
    esac
done

# --- Import container env (KOALA_USER, AWS creds) if not already set ---
if [[ -z "${KOALA_USER:-}" ]] && [[ -f /proc/1/environ ]]; then
    eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(AWS_|KOALA_|S3_)' | sed 's/^/export /')"
fi

# --- Env vars ---
export UV_FROZEN=1
export UV_PROJECT_ENVIRONMENT=/tmp/uv-venv
export HF_HOME=/local-ssd/hf_cache
export HF_HUB_DISABLE_XET=1
export RLLM_HOME=/local-ssd/rllm_home
mkdir -p /local-ssd/hf_cache /local-ssd/rllm_home

# --- Path conventions ---
PROJECT_DIR="/data/work/rllm"
S3_PREFIX="s3://arcwm-code-us-west-2/${KOALA_USER:-$USER}"
VENV_PATH="/tmp/uv-venv"
OUTPUT_LOCAL="/local-ssd/rllm-output"

# S3 cache paths
VENV_TAR_S3="${S3_PREFIX}/tools/rllm-venv.tar"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
HF_MODEL_SHORT=$(echo "${HF_MODEL}" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')
HF_TAR_S3="${S3_PREFIX}/tools/hf_cache_${HF_MODEL_SHORT}.tar"
DATASETS_TAR_S3="${S3_PREFIX}/tools/rllm-datasets-${ENV_NAME}.tar"

# ============================================================================
# Upload mode
# ============================================================================
if [[ ${UPLOAD_CACHE} -eq 1 ]]; then
    echo "=== Uploading verified environment to S3 ==="

    if [[ -d "${VENV_PATH}" ]]; then
        echo "  [1/3] Tarring venv -> ${VENV_TAR_S3}"
        tar cf - -C /tmp uv-venv | s5cmd pipe "${VENV_TAR_S3}"
        echo "  Done."
    else
        echo "  [1/3] SKIP: ${VENV_PATH} not found"
    fi

    if [[ -d "${HF_HOME}/hub" ]]; then
        echo "  [2/3] Tarring HF cache -> ${HF_TAR_S3}"
        tar cf - -C /local-ssd hf_cache | s5cmd pipe "${HF_TAR_S3}"
        echo "  Done."
    else
        echo "  [2/3] SKIP: no model in ${HF_HOME}"
    fi

    if [[ -d "${RLLM_HOME}/datasets" ]]; then
        echo "  [3/3] Tarring datasets -> ${DATASETS_TAR_S3}"
        tar cf - -C /local-ssd rllm_home | s5cmd pipe "${DATASETS_TAR_S3}"
        echo "  Done."
    else
        echo "  [3/3] SKIP: no datasets in ${RLLM_HOME}"
    fi

    echo "=== Upload complete ==="
    return 0 2>/dev/null || exit 0
fi

# ============================================================================
# Restore mode (default)
# ============================================================================

restore_venv() {
    echo "=== [1/4] Python venv ==="
    if [[ -x "${VENV_PATH}/bin/python" ]] && [[ ${FAST_MODE} -eq 1 ]]; then
        echo "  Already present (--fast), skip"
        return
    fi

    if s5cmd ls "${VENV_TAR_S3}" &>/dev/null; then
        echo "  Restoring from S3 cache..."
        s5cmd cat "${VENV_TAR_S3}" | tar xf - -C /tmp/
        echo "  Restored ($(du -sh ${VENV_PATH} | cut -f1))"
    else
        echo "  No S3 cache. Running uv sync (slow, ~10-15 min)..."
        cd "${PROJECT_DIR}"
        UV_FROZEN=0 uv sync --extra verl --extra rewards
        # verl 0.7.1 pulls tensordict 0.8.x but needs >=0.10 at runtime
        uv pip install --python "${VENV_PATH}/bin/python" 'tensordict>=0.10'
    fi
}

restore_hf_cache() {
    echo "=== [3/4] HF model cache ==="
    if [[ -d "${HF_HOME}/hub" ]] && [[ ${FAST_MODE} -eq 1 ]]; then
        echo "  Already present (--fast), skip"
        return
    fi

    if s5cmd ls "${HF_TAR_S3}" &>/dev/null; then
        echo "  Restoring from S3..."
        s5cmd cat "${HF_TAR_S3}" | tar xf - -C /local-ssd/
        echo "  Restored ($(du -sh ${HF_HOME} | cut -f1))"
    else
        echo "  No S3 cache, will download from HF on first use"
    fi
}

_default_env_setup() {
    local PIP_PYTHON="--python ${VENV_PATH}/bin/python"

    if [[ -n "${COOKBOOK_DIR:-}" ]]; then
        echo "  Installing cookbook: ${COOKBOOK_DIR}"
        uv pip install ${PIP_PYTHON} --no-deps -e "${PROJECT_DIR}/${COOKBOOK_DIR}"
    fi

    if [[ -n "${EXTRA_DEPS:-}" ]]; then
        echo "  Installing extra deps: ${EXTRA_DEPS}"
        read -ra _deps <<< "${EXTRA_DEPS}"
        uv pip install ${PIP_PYTHON} "${_deps[@]}"
    fi

    if [[ -n "${EXTRA_APT:-}" ]]; then
        echo "  Installing system packages: ${EXTRA_APT}"
        apt-get update -qq && apt-get install -y -qq ${EXTRA_APT}
    fi

    if [[ -n "${DATASETS:-}" ]]; then
        if s5cmd ls "${DATASETS_TAR_S3}" &>/dev/null; then
            echo "  Restoring datasets from S3..."
            s5cmd cat "${DATASETS_TAR_S3}" | tar xf - -C /local-ssd/
        else
            echo "  Pulling datasets from HuggingFace..."
            for ds in ${DATASETS}; do
                "${VENV_PATH}/bin/rllm" dataset pull "${ds}"
            done
        fi
    fi

    if [[ -n "${PREPARE_CMD:-}" ]]; then
        echo "  Running prepare: ${PREPARE_CMD}"
        (cd "${PROJECT_DIR}" && "${VENV_PATH}/bin/python" ${PREPARE_CMD})
    fi
}

run_env_setup() {
    echo "=== [2/4] Environment: ${ENV_NAME} ==="
    local ENV_SCRIPT="${PROJECT_DIR}/scripts/envs/${ENV_NAME}.sh"
    if [[ ! -f "${ENV_SCRIPT}" ]]; then
        echo "ERROR: env plugin not found: ${ENV_SCRIPT}"
        return 1 2>/dev/null || exit 1
    fi
    source "${ENV_SCRIPT}"

    if type env_setup &>/dev/null; then
        env_setup
    else
        _default_env_setup
    fi
}

setup_s3_sync() {
    echo "=== [4/4] Background S3 sync ==="
    if [[ -z "${EXP_NAME:-}" ]]; then
        echo "  WARN: EXP_NAME not set, skipping background sync"
        return
    fi
    local OUTPUT_S3="${S3_PREFIX}/experiments/${EXP_NAME}/output"
    mkdir -p "${OUTPUT_LOCAL}"
    (while true; do
        aws s3 sync "${OUTPUT_LOCAL}/" "${OUTPUT_S3}/" --delete --quiet 2>/dev/null || true
        sleep 300
    done) &
    local SYNC_PID=$!
    echo "  PID=${SYNC_PID} syncing to ${OUTPUT_S3} every 5 min"
    # Chain-merge with any pre-existing EXIT trap; sed assumes no embedded single-quotes in the prior body
    local _prev_trap
    _prev_trap=$(trap -p EXIT | sed "s/trap -- '\\(.*\\)' EXIT/\\1/" 2>/dev/null || true)
    trap "${_prev_trap:+${_prev_trap}; }kill ${SYNC_PID} 2>/dev/null || true; aws s3 sync ${OUTPUT_LOCAL}/ ${OUTPUT_S3}/ --quiet 2>/dev/null || true" EXIT TERM INT
}

_default_verify() {
    echo "=== Verify ==="
    local PY="${VENV_PATH}/bin/python"
    local ok=1
    ${PY} -c "import rllm; print('  rllm OK')" || ok=0
    ${PY} -c "import vllm; print(f'  vllm {vllm.__version__}')" || ok=0
    ${PY} -c "import verl; print('  verl OK')" || ok=0
    ${PY} -c "import torch; assert torch.cuda.is_available(); print(f'  torch {torch.__version__} cuda OK')" || ok=0
    if [[ $ok -eq 1 ]]; then
        echo "  All checks passed"
    else
        echo "  WARNING: some checks failed"
    fi
}

# --- Main ---
cd "${PROJECT_DIR}"

restore_venv
run_env_setup
restore_hf_cache
setup_s3_sync

export PATH="${VENV_PATH}/bin:${PATH}"

if type env_verify &>/dev/null; then
    env_verify
else
    _default_verify
fi

echo "=== Setup complete (env=${ENV_NAME}) ==="

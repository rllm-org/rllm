#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RLLM_ROOT="$(cd "$COOKBOOK_DIR/../.." && pwd)"
# shellcheck source=swe_artifact_utils.sh
source "$SCRIPT_DIR/swe_artifact_utils.sh"

export PYTHONPATH="$COOKBOOK_DIR:$RLLM_ROOT:$RLLM_ROOT/rllm-model-gateway/src:${PYTHONPATH:-}"
restore_swe_artifacts
exec /tmp/verl_venv/bin/python "$@"

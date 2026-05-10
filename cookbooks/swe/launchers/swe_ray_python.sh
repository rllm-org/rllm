#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=swe_artifact_utils.sh
source "$SCRIPT_DIR/swe_artifact_utils.sh"

restore_swe_artifacts
exec /tmp/verl_venv/bin/python "$@"

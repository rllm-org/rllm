#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK_ROOT="${WORK_ROOT:-/home/yanan/.deepcoder-sft-pipeline-formal}"
RUN_BASE="${RUN_BASE:-base-train-full-k8}"
DATA_DIR="${DATA_DIR:-$WORK_ROOT/curated/${RUN_BASE}-sft-data}"

export WORK_ROOT RUN_BASE DATA_DIR
MAX_EXAMPLES=all "$HERE/run_eval_dual.sh"
"$HERE/run_curate_dual.sh"

echo "Eval and curation complete: $DATA_DIR"
echo "SFT is intentionally not launched by the formal .16 collector."

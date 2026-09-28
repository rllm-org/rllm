#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$HERE/.." && pwd)"
source "$WORKSPACE/finqa-grpo-run/env.sh"
WORK_ROOT="${WORK_ROOT:-/home/yanan/.deepcoder-sft-pipeline-formal}"
RUN_BASE="${RUN_BASE:-base-train-full-k8}"
MAX_EXAMPLES="${MAX_EXAMPLES:-all}"

run_shard() {
  local shard="$1" gpu="$2" port="$3"
  GPU="$gpu" PORT="$port" NUM_SHARDS=2 SHARD_INDEX="$shard" \
    MAX_EXAMPLES="$MAX_EXAMPLES" WORK_ROOT="$WORK_ROOT" \
    RUN_NAME="${RUN_BASE}-shard${shard}" \
    "$HERE/run_eval.sh"
}

run_shard 0 0 8992 &
pid0=$!
run_shard 1 1 8993 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [[ "$status" != 0 ]]; then
  echo "one or more eval shards failed" >&2
  exit 1
fi

args=(
  --run-dir "$WORK_ROOT/eval_runs/${RUN_BASE}-shard0"
  --run-dir "$WORK_ROOT/eval_runs/${RUN_BASE}-shard1"
)
if [[ "$MAX_EXAMPLES" == all ]]; then args+=(--require-full); fi
"$VENV/bin/python" "$HERE/validate_shards.py" "${args[@]}" \
  | tee "$WORK_ROOT/${RUN_BASE}.shard_validation.json"

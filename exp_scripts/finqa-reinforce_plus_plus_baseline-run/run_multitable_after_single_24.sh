#!/usr/bin/env bash
# Start the RPP multi-table-v2 evaluation only after all requested single-table
# results have passed their exact row-count checks.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
V2_DIR="$(cd "$RUN_DIR/../finqa-multitable-v2" && pwd)"
SINGLE_ROOT="$RUN_DIR/eval_greedy_single"
RAW_ROOT="$RUN_DIR/checkpoints/qwen3-4b-24-rpp-n8-formal"
BASE_MODEL=/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
MERGED_CACHE=/home/yanan/.cache/finqa-rpp-multitable-v2-merged

complete_output() {
    python - "$1" "$2" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(1)
expected = int(sys.argv[2])
raise SystemExit(0 if data.get("total") == expected and len(data.get("items", [])) == expected else 1)
PY
}

# Do not start multi-table if either single-table lane failed or disappeared
# before producing all six complete outputs.
while kill -0 535029 2>/dev/null || kill -0 535030 2>/dev/null; do sleep 30; done
for tag in prpo_base prpo_global_step_125 prpo_global_step_250; do
    complete_output "$SINGLE_ROOT/$tag/val.json" 522
    complete_output "$SINGLE_ROOT/$tag/test.json" 558
done

common=(
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    FINQA_MULTITABLE_RUNTIME_TAG=rpp24
    FINQA_MULTITABLE_BASE_MODEL="$BASE_MODEL"
    FINQA_MULTITABLE_MERGED_CACHE="$MERGED_CACHE"
    FINQA_RPP_RAW_ROOT="$RAW_ROOT"
    FINQA_FORCE_KILL9=1
)

env "${common[@]}" bash "$V2_DIR/eval_lane.sh" rpp 0 8930 base global_step_250 &
lane0=$!
env "${common[@]}" bash "$V2_DIR/eval_lane.sh" rpp 1 8931 global_step_125 &
lane1=$!
wait "$lane0" "$lane1"
echo RPP_SINGLE_AND_MULTITABLE_COMPLETE

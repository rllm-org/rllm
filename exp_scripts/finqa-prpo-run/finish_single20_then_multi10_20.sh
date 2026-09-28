#!/usr/bin/env bash
# Bounded evaluation plan: finish single-table epochs 18-20, then evaluate
# multi-table epochs 10-20 on two local GPUs. Existing epoch-18/19 children may
# already be running when this coordinator starts, so it waits rather than
# launching duplicates.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_ROOT="$RUN_DIR/checkpoints/qwen3-4b-16-prpo-sync-b64-formal"
EVAL_ROOT="$RUN_DIR/eval_greedy"

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

# Epochs 18 and 19 were already dispatched by the old persistent watchers.
for step in 1116 1178; do
  while ! complete_output "$EVAL_ROOT/prpo_global_step_${step}/val.json" 522 ||
        ! complete_output "$EVAL_ROOT/prpo_global_step_${step}/test.json" 558; do
    sleep 60
  done
done

# Wait for the training process to atomically commit the epoch-20 checkpoint.
while [[ ! -d "$RAW_ROOT/global_step_1240/actor" ]]; do sleep 60; done

PRPO_RAW_ROOT="$RAW_ROOT" \
PRPO_MERGED_ROOT=/mnt/disk1t/finqa-prpo-run-checkpoints/single-eval-cache \
PRPO_EVAL_ROOT="$EVAL_ROOT" \
PRPO_RUNTIME_ROOT="$RUN_DIR/runtime_eval_post608/lane0" \
  bash "$RUN_DIR/eval_checkpoint_lane.sh" 0 8410 global_step_1240

complete_output "$EVAL_ROOT/prpo_global_step_1240/val.json" 522
complete_output "$EVAL_ROOT/prpo_global_step_1240/test.json" 558
find /mnt/disk1t/finqa-prpo-run-checkpoints/single-eval-cache/global_step_1240 -depth -delete 2>/dev/null || true

# Even and odd epochs use independent GPUs, ports, runtime homes and outputs.
bash "$RUN_DIR/eval_multitable_lane.sh" 0 8510 \
  global_step_620 global_step_744 global_step_868 \
  global_step_992 global_step_1116 global_step_1240 &
lane0=$!
bash "$RUN_DIR/eval_multitable_lane.sh" 1 8511 \
  global_step_682 global_step_806 global_step_930 \
  global_step_1054 global_step_1178 &
lane1=$!

wait "$lane0"
wait "$lane1"
echo "PRPO_EVAL_COMPLETE single-through-20 multi-10-through-20"

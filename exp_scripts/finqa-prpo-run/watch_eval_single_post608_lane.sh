#!/usr/bin/env bash
# Continuously evaluate complete post-step-608 epoch checkpoints. GPU 0 owns
# even epochs and GPU 1 owns odd epochs. The checkpoint tracker is the commit
# signal, so a directory still being written is never consumed.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU="${1:?usage: $0 GPU PORT}"
PORT="${2:?usage: $0 GPU PORT}"
[[ "$GPU" == 0 || "$GPU" == 1 ]] || { echo "GPU must be 0 or 1" >&2; exit 2; }

RAW_ROOT="$RUN_DIR/checkpoints/qwen3-4b-16-prpo-sync-b64-formal"
# Merged HF weights are disposable evaluation cache: the raw verl checkpoint
# remains authoritative. Keeping 41 additional 8 GB merges would exhaust this
# disk before epoch 50, so remove each merge only after both splits validate.
MERGED_ROOT=/mnt/disk1t/finqa-prpo-run-checkpoints/single-eval-cache
EVAL_ROOT="$RUN_DIR/eval_greedy"
TRACKER="$RAW_ROOT/latest_checkpointed_iteration.txt"
FINAL_STEP=3100

complete_output() {
  python - "$1" "$2" <<'PY'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
try:
    data = json.load(open(path))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if data.get("total") == expected and len(data.get("items", [])) == expected else 1)
PY
}

while true; do
  latest=0
  if [[ -s "$TRACKER" ]]; then latest="$(tr -cd '0-9' < "$TRACKER")"; fi
  did_work=0

  for ((step=620; step<=latest; step+=62)); do
    epoch=$((step / 62))
    (( epoch % 2 == GPU )) || continue
    val="$EVAL_ROOT/prpo_global_step_${step}/val.json"
    test="$EVAL_ROOT/prpo_global_step_${step}/test.json"
    if complete_output "$val" 522 && complete_output "$test" 558; then
      continue
    fi
    raw="$RAW_ROOT/global_step_${step}"
    [[ -d "$raw/actor" ]] || continue
    echo "[watch gpu=$GPU] evaluating epoch=$epoch global_step=$step tracker=$latest"
    PRPO_RAW_ROOT="$RAW_ROOT" \
    PRPO_MERGED_ROOT="$MERGED_ROOT" \
    PRPO_EVAL_ROOT="$EVAL_ROOT" \
    PRPO_RUNTIME_ROOT="$RUN_DIR/runtime_eval_post608/lane${GPU}" \
      bash "$RUN_DIR/eval_checkpoint_lane.sh" "$GPU" "$PORT" "global_step_${step}"
    merged="$MERGED_ROOT/global_step_${step}"
    if [[ "$merged" == /mnt/disk1t/finqa-prpo-run-checkpoints/single-eval-cache/global_step_* ]] && \
       complete_output "$val" 522 && complete_output "$test" 558 && [[ -d "$merged" ]]; then
      find "$merged" -depth -delete
      echo "[watch gpu=$GPU] removed reproducible merged cache for global_step=$step"
    fi
    did_work=1
  done

  if (( latest >= FINAL_STEP )); then
    pending=0
    for ((step=620; step<=FINAL_STEP; step+=62)); do
      epoch=$((step / 62))
      (( epoch % 2 == GPU )) || continue
      complete_output "$EVAL_ROOT/prpo_global_step_${step}/val.json" 522 || pending=1
      complete_output "$EVAL_ROOT/prpo_global_step_${step}/test.json" 558 || pending=1
    done
    (( pending == 0 )) && { echo "WATCH_COMPLETE gpu=$GPU through step=$FINAL_STEP"; exit 0; }
  fi

  (( did_work == 1 )) || sleep 300
done

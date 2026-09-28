#!/usr/bin/env bash
# Fill the local staging cache ahead of the two GPU evaluation lanes.
set -euo pipefail
ROOT=/mnt/disk1t/deepcoder-prpo-checkpoint-eval/raw16
REMOTE=/home/yanan/.deepcoder-checkpoints/deepcoder-prpo-b64-n1-16-c64-ni0-save10-ep3
mkdir -p "$ROOT"

# Give the active GPU-1 lane exclusive bandwidth for its first checkpoint so
# prefetching cannot delay the moment that the currently idle GPU starts.
while [[ ! -f "$ROOT/global_step_10/actor/lora_train_meta.json" ]]; do
  sleep 15
done

for STEP in $(seq 20 10 170); do
  DST="$ROOT/global_step_$STEP"
  [[ ! -f "$DST/actor/lora_train_meta.json" ]] || continue
  # If a lane already started its own transfer, do not compete with it.
  [[ ! -e "${DST}.incoming" ]] || continue
  TMP="${DST}.prefetching"
  rm -rf -- "$TMP"
  mkdir -p "$TMP"
  echo "[prefetch] step $STEP"
  rsync -a --partial \
    "10.225.68.16:$REMOTE/global_step_$STEP/" "$TMP/"
  test -f "$TMP/actor/lora_train_meta.json"
  mv "$TMP" "$DST"
done

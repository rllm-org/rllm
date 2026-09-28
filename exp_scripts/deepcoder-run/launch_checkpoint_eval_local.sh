#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOGROOT=/mnt/disk1t/deepcoder-prpo-checkpoint-eval/lane-logs
mkdir -p "$LOGROOT"

# Balance nine full evaluations per GPU.  host24 epoch 1 is included first.
setsid nohup bash "$ROOT/eval_checkpoint_lane.sh" 0 8990 \
  host24:379 host16:20 host16:40 host16:60 host16:80 \
  host16:100 host16:120 host16:140 host16:160 \
  >"$LOGROOT/gpu0.log" 2>&1 < /dev/null &
echo $! >"$LOGROOT/gpu0.pid"

setsid nohup bash "$ROOT/eval_checkpoint_lane.sh" 1 8991 \
  host16:10 host16:30 host16:50 host16:70 host16:90 \
  host16:110 host16:130 host16:150 host16:170 \
  >"$LOGROOT/gpu1.log" 2>&1 < /dev/null &
echo $! >"$LOGROOT/gpu1.pid"

echo "GPU0 PID $(cat "$LOGROOT/gpu0.pid")"
echo "GPU1 PID $(cat "$LOGROOT/gpu1.pid")"
echo "Results: /mnt/disk1t/deepcoder-prpo-checkpoint-eval/results"

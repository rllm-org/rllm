#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOGROOT=/mnt/disk1t/deepcoder-prpo-checkpoint-eval/lane-logs
P0="$(cat "$LOGROOT/gpu0.pid")"
P1="$(cat "$LOGROOT/gpu1.pid")"

setsid nohup bash "$ROOT/watch_new_checkpoints_local.sh" 0 8990 "$P0" 0 \
  >"$LOGROOT/watch-gpu0.log" 2>&1 < /dev/null &
echo $! >"$LOGROOT/watch-gpu0.pid"

setsid nohup bash "$ROOT/watch_new_checkpoints_local.sh" 1 8991 "$P1" 10 \
  >"$LOGROOT/watch-gpu1.log" 2>&1 < /dev/null &
echo $! >"$LOGROOT/watch-gpu1.pid"

echo "watch GPU0 PID $(cat "$LOGROOT/watch-gpu0.pid") waits for $P0"
echo "watch GPU1 PID $(cat "$LOGROOT/watch-gpu1.pid") waits for $P1"

#!/usr/bin/env bash
# Launch persistent local dual-GPU evaluation watchers for every complete
# PRPO epoch checkpoint after global_step_608.
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RUN_DIR/logs"

setsid nohup bash "$RUN_DIR/watch_eval_single_post608_lane.sh" 0 8410 \
  >"$RUN_DIR/logs/eval_single_post608_lane0.log" 2>&1 < /dev/null &
PID0=$!
printf '%s\n' "$PID0" >"$RUN_DIR/logs/eval_single_post608_lane0.pid"

setsid nohup bash "$RUN_DIR/watch_eval_single_post608_lane.sh" 1 8411 \
  >"$RUN_DIR/logs/eval_single_post608_lane1.log" 2>&1 < /dev/null &
PID1=$!
printf '%s\n' "$PID1" >"$RUN_DIR/logs/eval_single_post608_lane1.pid"

printf 'single-table watchers launched: gpu0 pid=%s port=8410; gpu1 pid=%s port=8411\n' "$PID0" "$PID1"

#!/usr/bin/env bash
# Two independent local GPU queues: one resident model per GPU, val then test.
set -euo pipefail
RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RUN_DIR/logs"

setsid nohup bash "$RUN_DIR/eval_checkpoint_lane.sh" 0 8210 \
  base global_step_32 global_step_64 global_step_96 \
  >"$RUN_DIR/logs/eval_greedy_lane0.log" 2>&1 < /dev/null &
pid0=$!
printf '%s\n' "$pid0" >"$RUN_DIR/logs/eval_greedy_lane0.pid"

setsid nohup bash "$RUN_DIR/eval_checkpoint_lane.sh" 1 8211 \
  global_step_16 global_step_48 global_step_80 \
  >"$RUN_DIR/logs/eval_greedy_lane1.log" 2>&1 < /dev/null &
pid1=$!
printf '%s\n' "$pid1" >"$RUN_DIR/logs/eval_greedy_lane1.pid"

printf 'lane0 pid=%s gpu=0 port=8210\nlane1 pid=%s gpu=1 port=8211\n' "$pid0" "$pid1"

#!/usr/bin/env bash
# Two local GPU queues. Requested near-integer anchors run first, followed by
# the remaining checkpoints already present in the PRPO result table.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RUN_DIR/logs"

LANE0=(
  base global_step_128 global_step_368 global_step_560
  global_step_240 global_step_272 global_step_320 global_step_352 global_step_400
  global_step_432 global_step_464 global_step_512 global_step_544 global_step_592
)
LANE1=(
  global_step_64 global_step_304 global_step_496 global_step_608
  global_step_192 global_step_256 global_step_288 global_step_336 global_step_384
  global_step_416 global_step_448 global_step_480 global_step_528 global_step_576
)

setsid nohup bash "$RUN_DIR/eval_multitable_lane.sh" 0 8310 "${LANE0[@]}" \
  >"$RUN_DIR/logs/eval_multitable_prpo_lane0.log" 2>&1 < /dev/null &
PID0=$!
printf '%s\n' "$PID0" >"$RUN_DIR/logs/eval_multitable_prpo_lane0.pid"

setsid nohup bash "$RUN_DIR/eval_multitable_lane.sh" 1 8311 "${LANE1[@]}" \
  >"$RUN_DIR/logs/eval_multitable_prpo_lane1.log" 2>&1 < /dev/null &
PID1=$!
printf '%s\n' "$PID1" >"$RUN_DIR/logs/eval_multitable_prpo_lane1.pid"

printf 'lane0 pid=%s gpu=0 port=8310\nlane1 pid=%s gpu=1 port=8311\n' "$PID0" "$PID1"

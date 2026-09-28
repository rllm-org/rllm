#!/usr/bin/env bash
# Launch the fixed base + 21-checkpoint multi-table evaluation queue on .16.
set -euo pipefail

RD="$(cd "$(dirname "$0")" && pwd)"
HOST="$(hostname -I 2>/dev/null | awk '{print $1}')"
[ "$HOST" = "10.225.68.16" ] || {
    echo "refusing to launch: this queue is reserved for .16 (host IP is $HOST)" >&2
    exit 1
}

LANE0=(
    base
    global_step_93 global_step_155 global_step_217 global_step_279 global_step_341
    global_step_403 global_step_465 global_step_527 global_step_589 global_step_651
)
LANE1=(
    global_step_124 global_step_186 global_step_248 global_step_310 global_step_372
    global_step_434 global_step_496 global_step_558 global_step_620 global_step_682
    global_step_713
)

mkdir -p "$RD/logs"
nohup bash "$RD/eval_multitable_lane.sh" 0 8120 "${LANE0[@]}" \
    > "$RD/logs/eval_multitable_lane0.log" 2>&1 < /dev/null &
PID0=$!
nohup bash "$RD/eval_multitable_lane.sh" 1 8121 "${LANE1[@]}" \
    > "$RD/logs/eval_multitable_lane1.log" 2>&1 < /dev/null &
PID1=$!
printf '%s\n' "$PID0" > "$RD/logs/eval_multitable_lane0.pid"
printf '%s\n' "$PID1" > "$RD/logs/eval_multitable_lane1.pid"
echo "multi-table lanes launched: gpu0 pid=$PID0 gpu1 pid=$PID1"

#!/usr/bin/env bash
# Launch two independent local GPU queues. Existing v2 output is never mixed.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR/logs"

for pidfile in "$DIR/logs"/lane*.pid; do
    [[ -e "$pidfile" ]] || continue
    old_pid="$(cat "$pidfile")"
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "existing v2 lane still running: pid=$old_pid ($pidfile)" >&2
        exit 1
    fi
done

setsid nohup bash "$DIR/run_lane_plan.sh" 0 8620 >"$DIR/logs/lane0.log" 2>&1 < /dev/null &
pid0=$!
printf '%s\n' "$pid0" >"$DIR/logs/lane0.pid"

setsid nohup bash "$DIR/run_lane_plan.sh" 1 8621 >"$DIR/logs/lane1.log" 2>&1 < /dev/null &
pid1=$!
printf '%s\n' "$pid1" >"$DIR/logs/lane1.pid"

printf 'multi-table-v2 lane0 pid=%s gpu=0 port=8620\n' "$pid0"
printf 'multi-table-v2 lane1 pid=%s gpu=1 port=8621\n' "$pid1"

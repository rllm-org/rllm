#!/usr/bin/env bash
# Read-only recurring GPU/error/progress snapshots. Never restarts a failed arm.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
while true; do
  date -Is
  for arm in grpo prpo; do
    host=10.225.68.16; [[ "$arm" == prpo ]] && host=10.225.68.24
    echo "$arm $host"
    ssh -o ConnectTimeout=10 "$host" 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
    grep -E 'step:[0-9]+|OutOfMemoryError|CUDA out of memory|Traceback|Watchdog caught|Error executing job|ValueError|RuntimeError' "$ROOT/logs/$arm.log" | tail -3
  done
  sleep 60
done

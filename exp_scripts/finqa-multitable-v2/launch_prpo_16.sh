#!/usr/bin/env bash
# Launch the PRPO-only multi-table-v2 queues on 10.225.68.16.
set -euo pipefail

HOST=10.225.68.16
DIR=/home/yanan/agents/rllm/exp_scripts/finqa-multitable-v2

ssh "$HOST" "mkdir -p '$DIR/logs'; \
  setsid -f env \
    FINQA_MULTITABLE_MERGED_CACHE=/home/yanan/.cache/finqa-multitable-v2-merged-cache \
    FINQA_MULTITABLE_RUNTIME_TAG=tai \
    bash '$DIR/run_prpo_lane.sh' 0 8720 >'$DIR/logs/prpo16_lane0.log' 2>&1 </dev/null"
ssh "$HOST" "setsid -f env \
    FINQA_MULTITABLE_MERGED_CACHE=/home/yanan/.cache/finqa-multitable-v2-merged-cache \
    FINQA_MULTITABLE_RUNTIME_TAG=tai \
    bash '$DIR/run_prpo_lane.sh' 1 8721 >'$DIR/logs/prpo16_lane1.log' 2>&1 </dev/null"

sleep 1
pid0="$(ssh "$HOST" "pgrep -fo 'bash $DIR/run_prpo_lane.sh 0 8720'")"
pid1="$(ssh "$HOST" "pgrep -fo 'bash $DIR/run_prpo_lane.sh 1 8721'")"
printf '%s\n' "$pid0" >"$DIR/logs/prpo16_lane0.pid"
printf '%s\n' "$pid1" >"$DIR/logs/prpo16_lane1.pid"
printf '.16 PRPO lane0 pid=%s gpu=0 port=8720\n' "$pid0"
printf '.16 PRPO lane1 pid=%s gpu=1 port=8721\n' "$pid1"

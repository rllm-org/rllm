#!/usr/bin/env bash
# Windowed: alert on NEW real timeouts since this watcher started (>= 3 new), else report after MAXMIN.
W=/home/yanan/agents/rllm/exp_scripts/deepcoder-run/grader_ab/watch_grpo24; MAXMIN=${1:-60}
L0=$(bash $W/check.sh); a0=$(sed -E 's/.*alarms=([0-9]+).*/\1/' <<<"$L0"); n0=$(sed -E 's/.*rollouts=([0-9]+).*/\1/' <<<"$L0")
for i in $(seq 1 $((MAXMIN/3))); do
  sleep 180; L=$(bash $W/check.sh); echo "$(date +%T) $L" >> $W/history.txt
  a=$(sed -E 's/.*alarms=([0-9]+).*/\1/' <<<"$L"); n=$(sed -E 's/.*rollouts=([0-9]+).*/\1/' <<<"$L")
  if [ $((a-a0)) -ge 3 ]; then echo "ALERT $(date -Is): $((a-a0)) new timeouts in $((n-n0)) new rollouts -> $L"; exit 2; fi
done
echo "REPORT $(date -Is): $((a-a0)) new timeouts in $((n-n0)) new rollouts over ${MAXMIN} min -> $L"

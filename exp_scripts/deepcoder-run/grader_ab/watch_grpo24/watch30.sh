#!/usr/bin/env bash
# 30-minute cadence report on grader alarms for the current .24 run; exits early only on >= 3 NEW real timeouts.
W=/home/yanan/agents/rllm/exp_scripts/deepcoder-run/grader_ab/watch_grpo24
L0=$(bash $W/check_p4ni0.sh); a0=$(sed -E 's/.*alarms=([0-9]+).*/\1/' <<<"$L0"); n0=$(sed -E 's/.*rollouts=([0-9]+).*/\1/' <<<"$L0")
for i in $(seq 1 10); do
  sleep 180; L=$(bash $W/check_p4ni0.sh); echo "$(date +%T) $L" >> $W/history_p4ni0.txt
  a=$(sed -E 's/.*alarms=([0-9]+).*/\1/' <<<"$L"); n=$(sed -E 's/.*rollouts=([0-9]+).*/\1/' <<<"$L")
  if [ $((a-a0)) -ge 3 ]; then echo "ALERT $(date -Is): $((a-a0)) new real timeouts in $((n-n0)) new rollouts -> $L"; exit 2; fi
done
echo "REPORT $(date -Is): $((a-a0)) new real timeouts in $((n-n0)) new rollouts over 30 min -> $L"

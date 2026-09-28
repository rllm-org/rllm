#!/usr/bin/env bash
# 2-hour cadence report on grader alarms for the current .24 run; exits early only on >= 3 NEW real timeouts.
W=/home/yanan/agents/rllm/exp_scripts/deepcoder-run/grader_ab/watch_grpo24
L0=$(bash $W/check_p4ni0.sh); a0=$(sed -E 's/.*alarms=([0-9]+).*/\1/' <<<"$L0"); n0=$(sed -E 's/.*rollouts=([0-9]+).*/\1/' <<<"$L0")
for i in $(seq 1 40); do
  sleep 180; L=$(bash $W/check_p4ni0.sh); echo "$(date +%T) $L" >> $W/history_p4ni0.txt
  a=$(sed -E 's/.*alarms=([0-9]+).*/\1/' <<<"$L"); n=$(sed -E 's/.*rollouts=([0-9]+).*/\1/' <<<"$L")
  # Early exit only on a regime change (the ~1% steady state is known): >= 5% new-alarm rate over >= 200 new rollouts.
  if [ $((n-n0)) -ge 200 ] && awk "BEGIN{exit !(($a-$a0)/($n-$n0) >= 0.05)}"; then echo "ALERT $(date -Is): $((a-a0)) new real timeouts in $((n-n0)) new rollouts (>= 5%) -> $L"; exit 2; fi
done
echo "REPORT $(date -Is): $((a-a0)) new real timeouts in $((n-n0)) new rollouts over 2 h -> $L"

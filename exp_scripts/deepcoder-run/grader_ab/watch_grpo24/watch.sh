#!/usr/bin/env bash
# Poll every 3 min. Exit immediately (so the session is notified) when the .24 timeout signature appears:
#   >= 3 real timeouts, or alarm rate >= 0.5% with >= 200 rollouts, or eval>=12s share >= 3% with >= 200 rollouts.
# Otherwise exit after MAXMIN minutes with a normal report.
W=/home/yanan/agents/rllm/exp_scripts/deepcoder-run/grader_ab/watch_grpo24; MAXMIN=${1:-90}
for i in $(seq 1 $((MAXMIN/3))); do
  L=$(bash $W/check.sh); echo "$(date +%T) $L" >> $W/history.txt
  n=$(sed -E 's/.*rollouts=([0-9]+).*/\1/' <<<"$L"); a=$(sed -E 's/.*alarms=([0-9]+).*/\1/' <<<"$L")
  ar=$(sed -E 's/.*alarm_rate=([0-9.]+)%.*/\1/' <<<"$L"); sl=$(sed -E 's/.*eval_ge12=[0-9]+ \(([0-9.]+)%\).*/\1/' <<<"$L")
  if [ "$a" -ge 3 ] || { [ "$n" -ge 200 ] && { awk "BEGIN{exit !($ar>=0.5)}" || awk "BEGIN{exit !($sl>=3)}"; }; }; then
    echo "ALERT $(date -Is): timeout signature detected -> $L"; exit 2; fi
  sleep 180
done
echo "OK $(date -Is): no timeout signature in ${MAXMIN} min -> $L"

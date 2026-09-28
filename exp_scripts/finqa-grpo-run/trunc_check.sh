#!/usr/bin/env bash
# Runs ON .24. Truncation / turn-exhaustion accounting for the FinQA Qwen3.5 run.
RD=/home/yanan/agents/rllm/exp_scripts/finqa-grpo-run/eval/qwen35-4b-base
for s in val test; do
  L="$RD/eval_$s.log"
  if [ ! -f "$L" ]; then echo "  $s: no log yet"; continue; fi
  done_n=$(grep -c "Rollout completed" "$L" 2>/dev/null)
  prog=$(grep -oE "[0-9]+/[0-9]+ \[[^]]*\]" "$L" | tail -1)
  echo "  $s: rollouts=$done_n progress=$prog"
  echo -n "    terminations: "
  grep -oE "TerminationReason\.[A-Z_]+" "$L" 2>/dev/null | sort | uniq -c | tr "\n" " "
  echo
  bad=$(grep -oE "TerminationReason\.(MAX_RESPONSE_LENGTH_EXCEEDED|MAX_PROMPT_LENGTH_EXCEEDED|MAX_TURNS_EXCEEDED|TIMEOUT|ERROR)" "$L" 2>/dev/null | wc -l)
  if [ "$done_n" -gt 0 ]; then
    pct=$(( 100 * bad / done_n ))
    echo "    BAD_TERMINATIONS: $bad/$done_n = ${pct}%"
    [ "$pct" -gt 5 ] && echo "    ALERT_TRUNCATION $s ${pct}%"
  fi
  V="$RD/vllm_$s.log"
  [ -f "$V" ] && echo "    server length-capped completions: $(grep -c 'finish_reason.*length' "$V" 2>/dev/null)"
  R="$RD/$s.json"
  [ -f "$R" ] && python3 -c "
import json;d=json.load(open('$R'))
print(f'    RESULT {100*d[\"correct\"]/d[\"total\"]:.2f}% ({d[\"correct\"]}/{d[\"total\"]})')" 2>/dev/null
done

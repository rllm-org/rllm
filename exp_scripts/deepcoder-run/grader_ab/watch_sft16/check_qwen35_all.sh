#!/usr/bin/env bash
# Qwen3.5-4B runs on both hosts, with explicit truncation / turn-exhaustion
# accounting. A run dominated by MAX_*_EXCEEDED reports a low score for the
# wrong reason, so those are alerted on (>5% of completed rollouts), not just logged.
echo "T=$(date '+%F %T %Z')"
echo "=== FinQA on .24 (val 522 / test 558) ==="
ssh -o ConnectTimeout=10 10.225.68.24 'bash /tmp/finqa_trunc_check.sh'
echo "=== DeepCoder on .29 (dual-sharded, 687 total, thinking off) ==="
D=/mnt/disk1t/deepcoder-prpo-checkpoint-eval/base-qwen35-dual
if [ -f "$D/result.json" ]; then
  python3 -c "import json;x=json.load(open('$D/result.json'));print(f'  MERGED {100*x[\"correct\"]/x[\"total\"]:.2f}% ({x[\"correct\"]}/{x[\"total\"]}) errors={x[\"errors\"]}')"
else
  for i in 0 1; do
    L="$D/shard$i.log"
    echo -n "  shard$i: "; grep -oE "[0-9]+/[0-9]+ \[[^]]*\]" "$L" 2>/dev/null | tail -1 || echo "(no progress)"
    echo -n "    terminations: "; grep -oE "TerminationReason\.[A-Z_]+" "$L" 2>/dev/null | sort | uniq -c | tr "\n" " "; echo
    bad=$(grep -oE "TerminationReason\.(MAX_RESPONSE_LENGTH_EXCEEDED|MAX_PROMPT_LENGTH_EXCEEDED|MAX_TURNS_EXCEEDED|TIMEOUT|ERROR)" "$L" 2>/dev/null | wc -l)
    tot=$(grep -c "Rollout completed" "$L" 2>/dev/null)
    if [ "${tot:-0}" -gt 0 ]; then pct=$(( 100 * bad / tot )); echo "    BAD_TERMINATIONS: $bad/$tot = ${pct}%"; [ "$pct" -gt 5 ] && echo "    ALERT_TRUNCATION deepcoder-shard$i ${pct}%"; fi
  done
fi
echo -n "GPU .29: "; nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo

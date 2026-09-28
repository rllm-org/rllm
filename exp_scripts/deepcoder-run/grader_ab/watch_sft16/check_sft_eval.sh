#!/usr/bin/env bash
# One read-only snapshot of the SFT checkpoint evaluation on .29.
W=/mnt/disk1t/deepcoder-prpo-checkpoint-eval
echo "T=$(date '+%F %T %Z')"
if pgrep -f run_sft_eval_all >/dev/null; then echo "orchestrator=alive"; else echo "orchestrator=GONE"; fi
echo "--- orchestrator log ---"; tail -6 "$W/lane-logs/sft-orchestrator.log"
echo "--- SFT checkpoints ---"
for d in "$W"/results-sft-c32/*/; do [ -d "$d" ] || continue; t=$(basename "$d")
  if [ -s "$d/result.json" ]; then
    python3 -c "import json;x=json.load(open('$d/result.json'));print(f'  {\"$t\":26s} DONE {100*x[\"correct\"]/x[\"total\"]:.2f}% ({x[\"correct\"]}/{x[\"total\"]})')"
  else echo "  $t  in progress"; fi
done
echo "--- base control ---"
for d in "$W"/base-sft-c32/*/; do [ -d "$d" ] || continue; t=$(basename "$d")
  if [ -s "$d/result.json" ]; then
    python3 -c "import json;x=json.load(open('$d/result.json'));print(f'  {\"$t\":20s} DONE {100*x[\"correct\"]/x[\"total\"]:.2f}% ({x[\"correct\"]}/{x[\"total\"]})')"
  else echo "  $t  in progress"; fi
done
echo -n "GPU: "; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' '; echo
echo -n "FAILS: "; grep -ci "vLLM exited before ready\|Traceback" "$W"/lane-logs/sft-gpu*.log 2>/dev/null | tr '\n' ' '; echo

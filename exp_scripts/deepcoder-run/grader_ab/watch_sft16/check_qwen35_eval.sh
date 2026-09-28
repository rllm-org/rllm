#!/usr/bin/env bash
W=/mnt/disk1t/deepcoder-prpo-checkpoint-eval
echo "T=$(date '+%F %T %Z')"
for d in think nothink; do
  R="$W/base-qwen35-$d-c32"
  echo "--- $d ---"
  if [ -s "$R"/gpu*-repeat1/result.json ]; then
    python3 -c "import json,glob;x=json.load(open(glob.glob('$R/gpu*-repeat1/result.json')[0]));print(f'  DONE {100*x[\"correct\"]/x[\"total\"]:.2f}% ({x[\"correct\"]}/{x[\"total\"]}) errors={x[\"errors\"]}')"
  else
    echo -n "  progress: "; tail -1 "$R"/gpu*-repeat1.log 2>/dev/null | grep -oE "[0-9]+/687 \[[^]]*\]" | tail -1 || echo "(no progress line)"
  fi
done
echo -n "GPU: "; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' '; echo
pgrep -f "eval_base_lane" >/dev/null && echo "lanes=alive" || echo "lanes=GONE"

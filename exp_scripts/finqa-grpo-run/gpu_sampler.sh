#!/usr/bin/env bash
# Sample per-GPU memory every 2s while a training run is alive; report the peak.
OUT="$1"; TARGET_PID="$2"
: > "$OUT"
while kill -0 "$TARGET_PID" 2>/dev/null; do
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "$OUT"
  sleep 2
done
echo "--- PEAK per GPU (MiB) ---" >> "$OUT"
awk -F', *' '{if($2>m[$1])m[$1]=$2} END{for(g in m) printf "  GPU %s peak %d MiB (%.2f GiB)\n", g, m[g], m[g]/1024}' "$OUT" >> "$OUT"

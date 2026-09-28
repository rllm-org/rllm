#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")" && pwd)"
sleep 1800
OUT=$(bash "$HERE/check_sft_eval.sh" 2>&1)
if echo "$OUT" | grep -q "orchestrator=GONE"; then echo "ALERT $(date '+%F %T %Z')"; else echo "REPORT $(date '+%F %T %Z')"; fi
echo "$OUT"

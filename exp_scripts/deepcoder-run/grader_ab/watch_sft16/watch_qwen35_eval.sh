#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")" && pwd)"
sleep 1800
OUT=$(bash "$HERE/check_qwen35_eval.sh" 2>&1)
if echo "$OUT" | grep -q "lanes=GONE"; then echo "ALERT $(date '+%F %T %Z')"; else echo "REPORT $(date '+%F %T %Z')"; fi
echo "$OUT"

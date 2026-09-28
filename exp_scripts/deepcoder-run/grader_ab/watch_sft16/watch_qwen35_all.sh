#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")" && pwd)"
sleep 1800
OUT=$(bash "$HERE/check_qwen35_all.sh" 2>&1)
if echo "$OUT" | grep -q "ALERT_TRUNCATION"; then echo "ALERT $(date '+%F %T %Z') truncation/turn-exhaustion above 5%"
else echo "REPORT $(date '+%F %T %Z')"; fi
echo "$OUT"

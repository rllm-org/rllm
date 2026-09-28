#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")" && pwd)"
sleep 1800
OUT=$(bash "$HERE/check_finqa_chain.sh" 2>&1)
if echo "$OUT" | grep -qE "chain=GONE|CHAIN ABORTED|ZERO SCORE|ALERT_CONTEXT|ALERT_TURN_EXHAUSTION"; then echo "ALERT $(date '+%F %T %Z')"; else echo "REPORT $(date '+%F %T %Z')"; fi
echo "$OUT"

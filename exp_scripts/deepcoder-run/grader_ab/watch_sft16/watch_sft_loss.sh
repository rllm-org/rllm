#!/usr/bin/env bash
# Sleep one 30-minute cycle, then emit one SFT loss report and exit.
# Short cadence because the whole run is only ~2 h.
HERE="$(cd "$(dirname "$0")" && pwd)"
sleep 1800
OUT=$(bash "$HERE/check_sft_loss.sh" 2>&1)
if echo "$OUT" | grep -qE '^FATAL|NAN_OR_INF=[1-9]|STATUS=COMPLETE'; then
  echo "ALERT $(date '+%F %T %Z')"
else
  echo "REPORT $(date '+%F %T %Z')"
fi
echo "$OUT"

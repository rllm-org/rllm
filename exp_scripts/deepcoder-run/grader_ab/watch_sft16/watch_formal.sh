#!/usr/bin/env bash
# 2-h cadence; early exit on ERRORS or dead drivers.
W=/home/yanan/agents/rllm/exp_scripts/deepcoder-run/grader_ab/watch_sft16
for i in $(seq 1 24); do
  sleep 300; S=$(bash $W/check_formal.sh); echo "$S" >> $W/history_formal.txt; echo ---- >> $W/history_formal.txt
  if grep -qE "^ERRORS|drivers=0" <<<"$S"; then echo "ALERT $(date -Is)"; echo "$S"; exit 2; fi
done
echo "REPORT $(date -Is)"; echo "$S"

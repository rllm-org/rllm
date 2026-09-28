#!/usr/bin/env bash
# 30-min cadence; early exit on Traceback or dead process.
W=/home/yanan/agents/rllm/exp_scripts/deepcoder-run/grader_ab/watch_grpo24
for i in $(seq 1 40); do
  sleep 180; S=$(bash $W/check_c64drop.sh); echo "$S" >> $W/history_c64drop.txt; echo ---- >> $W/history_c64drop.txt
  # Alert only on real failures: dead process, Hydra job error, Ray OOM kill. A traceback alone is
  # not fatal (the deepcoder flow logs httpx timeouts with a traceback and scores the rollout 0).
  if grep -qE "alive=NO|^FATAL" <<<"$S"; then echo "ALERT $(date -Is)"; echo "$S"; exit 2; fi
done
echo "REPORT $(date -Is)"; echo "$S"

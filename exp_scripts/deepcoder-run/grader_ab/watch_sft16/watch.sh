#!/usr/bin/env bash
# Poll every 5 min for up to 2 h. Exit early on any ERRORS line, or when an SFT checkpoint dir appears that
# was NOT present at watcher start (baseline snapshot), so the session is notified of new events only.
W=/home/yanan/agents/rllm/exp_scripts/deepcoder-run/grader_ab/watch_sft16
S0=$(bash $W/check.sh); BASE=$(grep -E "^SFT_CKPT" <<<"$S0" | sort)
for i in $(seq 1 24); do
  S=$(bash $W/check.sh); echo "$S" >> $W/history.txt; echo "----" >> $W/history.txt
  if grep -q "^ERRORS" <<<"$S"; then echo "ALERT $(date -Is)"; echo "$S"; exit 2; fi
  NEW=$(comm -13 <(echo "$BASE") <(grep -E "^SFT_CKPT" <<<"$S" | sort))
  if [ -n "$NEW" ]; then echo "EVENT $(date -Is): new SFT checkpoint state"; echo "$NEW"; echo "$S"; exit 3; fi
  sleep 300
done
echo "REPORT $(date -Is)"; echo "$S"

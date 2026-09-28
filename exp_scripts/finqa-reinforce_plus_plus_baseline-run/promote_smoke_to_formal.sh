#!/usr/bin/env bash
# Wait for a detached two-step smoke and promote it to the formal run only when
# the log contains the expected algorithm-level health evidence.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
SMOKE_PID_FILE="${1:-$RUN_DIR/logs/rpp-n8-smoke-v2.pid}"
SMOKE_LOG="${2:-$RUN_DIR/logs/rpp-n8-smoke-v2.log}"
FORMAL_LOG="${3:-$RUN_DIR/logs/rpp-n8-formal.log}"
PROMOTION_LOG="$RUN_DIR/logs/promotion.log"

smoke_pid="$(cat "$SMOKE_PID_FILE")"
while kill -0 "$smoke_pid" 2>/dev/null; do
  sleep 30
done

fail() {
  printf '%s promotion refused: %s\n' "$(date --iso-8601=seconds)" "$1" >> "$PROMOTION_LOG"
  exit 1
}

[[ -s "$SMOKE_LOG" ]] || fail "smoke log is empty"
[[ "$(rg -c '^.*step:[12] ' "$SMOKE_LOG" || true)" -eq 2 ]] || fail "did not observe exactly two completed optimiser steps"
[[ "$(rg -c 'groups/num_groups:32' "$SMOKE_LOG" || true)" -eq 2 ]] || fail "group count was not 32 on both steps"
[[ "$(rg -c 'groups/avg_group_size:np.float64\(8\.0\)' "$SMOKE_LOG" || true)" -eq 2 ]] || fail "group size was not 8 on both steps"
[[ "$(rg -c 'groups/num_trajs_after_filter:np.int64\(256\)' "$SMOKE_LOG" || true)" -eq 2 ]] || fail "did not retain 256 trajectories on both steps"
[[ "$(rg -c 'advantage/finqa/std:np.float64\(' "$SMOKE_LOG" || true)" -eq 2 ]] || fail "missing advantage statistics"
[[ "$(rg -c 'grad_norm:np.float64\(' "$SMOKE_LOG" || true)" -eq 2 ]] || fail "missing gradient statistics"
if rg -q 'OutOfMemoryError|CUDA out of memory|RayTaskError|NCCL.*(error|timeout)|[Nn]an.*grad_norm' "$SMOKE_LOG"; then
  fail "fatal signature found in smoke log"
fi

printf '%s smoke accepted; launching formal run\n' "$(date --iso-8601=seconds)" >> "$PROMOTION_LOG"
setsid nohup bash "$RUN_DIR/train_local_reinforce_plus_plus.sh" formal >"$FORMAL_LOG" 2>&1 < /dev/null &
formal_pid=$!
printf '%s\n' "$formal_pid" > "$RUN_DIR/logs/rpp-n8-formal.pid"
printf '%s formal pid=%s log=%s\n' "$(date --iso-8601=seconds)" "$formal_pid" "$FORMAL_LOG" >> "$PROMOTION_LOG"

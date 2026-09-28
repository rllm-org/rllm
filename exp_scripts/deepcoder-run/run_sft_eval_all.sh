#!/usr/bin/env bash
# Evaluate the four self-SFT checkpoints plus a fresh base control on the two
# local GPUs, against the DeepCoder *test* split (687 tasks) at concurrency 32.
#
# Protocol comes from eval_base.py and is asserted there, not configured here:
#   split=test, 687 tasks, temperature=0, top_p=1.0, seed=1234, max_tokens=16384
# EVAL_CONCURRENCY=32 matches the c32 base runs (166/687 twice) so the SFT
# numbers are directly comparable to the PRPO and GRPO tables.
#
# Staging is owned by prestage_sft.sh alone. Each worker WAITS for a checkpoint
# to be published before calling the lane, so the lane finds it already staged
# and never rsyncs -- two writers on the same .incoming path would race.
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
W=/mnt/disk1t/deepcoder-prpo-checkpoint-eval
export EVAL_CONCURRENCY=32
export RESULTS_DIR="$W/results-sft-c32"
LOGS="$W/lane-logs"
mkdir -p "$RESULTS_DIR" "$LOGS"

wait_staged() {  # $1 = step; block until prestage_sft.sh publishes it
  local step="$1" raw="$W/raw16/sft16_global_step_$1"
  for ((i=0; i<480; i++)); do
    [[ -f "$raw/actor/lora_train_meta.json" ]] && return 0
    sleep 30
  done
  echo "[wait_staged] TIMEOUT waiting for $raw" >&2
  return 1
}

lane_a() {  # GPU 0: the checkpoints that stage first and last
  for STEP in 136 408 544; do
    wait_staged "$STEP" || return 1
    echo "[lane_a] $(date '+%T') starting sft16:$STEP"
    bash "$ROOT/eval_checkpoint_lane.sh" 0 28110 "sft16:$STEP" >>"$LOGS/sft-gpu0.log" 2>&1
  done
  echo "[lane_a] $(date '+%T') done"
}

lane_b() {  # GPU 1: base control first (needs no staging), then the middle one
  echo "[lane_b] $(date '+%T') starting base control (2 repeats)"
  bash "$ROOT/eval_base_lane.sh" 1 28111 "$W/base-sft-c32" >>"$LOGS/sft-gpu1.log" 2>&1
  wait_staged 272 || return 1
  echo "[lane_b] $(date '+%T') starting sft16:272"
  bash "$ROOT/eval_checkpoint_lane.sh" 1 28111 "sft16:272" >>"$LOGS/sft-gpu1.log" 2>&1
  echo "[lane_b] $(date '+%T') done"
}

lane_a & A=$!
lane_b & B=$!
wait "$A"; echo "lane_a exit=$?"
wait "$B"; echo "lane_b exit=$?"
echo "ALL EVALS FINISHED $(date '+%F %T %Z')"

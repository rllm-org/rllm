#!/usr/bin/env bash
# Continuously evaluate complete DeepCoder GRPO checkpoints produced on .24.
#
# This is orchestration only.  It delegates each checkpoint to
# eval_checkpoint_lane.sh, whose eval_base.py stores aggregate result.json only
# and deliberately does not serialize returned episodes/trajectories.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REMOTE="${GRPO24_HOST:-10.225.68.24}"
REMOTE_ROOT="${GRPO24_REMOTE_ROOT:-/home/yanan/.deepcoder-checkpoints/deepcoder-grpo-b16-n8-24-c64-drop-save64}"
RESULTS_DIR="${RESULTS_DIR:-/mnt/disk1t/deepcoder-prpo-checkpoint-eval/results-grpo-c32}"
POLL_SECONDS="${POLL_SECONDS:-300}"
PORT0="${PORT0:-28110}"
PORT1="${PORT1:-28111}"
LOG_ROOT="/mnt/disk1t/deepcoder-prpo-checkpoint-eval/lane-logs"
mkdir -p "$RESULTS_DIR" "$LOG_ROOT"

# The initial manually scheduled lanes may still be staging or evaluating.
# Waiting here prevents two processes from using the same GPU or checkpoint.
for pid in "$@"; do
  if [[ "$pid" =~ ^[0-9]+$ ]]; then
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  fi
done

while :; do
  mapfile -t remote_steps < <(
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
      "find '$REMOTE_ROOT' -mindepth 3 -maxdepth 3 -path '*/actor/lora_train_meta.json' -printf '%h\n'" \
      | sed -n 's#.*/global_step_\([0-9][0-9]*\)/actor#\1#p' \
      | sort -n
  ) || { sleep "$POLL_SECONDS"; continue; }

  pending=()
  for step in "${remote_steps[@]}"; do
    [[ -s "$RESULTS_DIR/grpo24_global_step_${step}/result.json" ]] || pending+=("$step")
  done

  if ((${#pending[@]})); then
    # Evaluate newest checkpoints first.  Each lane is sequential internally;
    # the two lanes run concurrently on separate GPUs.
    mapfile -t pending < <(printf '%s\n' "${pending[@]}" | sort -nr)
    lane0=(); lane1=()
    for i in "${!pending[@]}"; do
      if ((i % 2 == 0)); then lane0+=("grpo24:${pending[$i]}");
      else lane1+=("grpo24:${pending[$i]}"); fi
    done

    pids=()
    if ((${#lane0[@]})); then
      RESULTS_DIR="$RESULTS_DIR" EVAL_CONCURRENCY=32 \
        bash "$ROOT/eval_checkpoint_lane.sh" 0 "$PORT0" "${lane0[@]}" \
        >>"$LOG_ROOT/grpo24-watch-gpu0.log" 2>&1 &
      pids+=("$!")
    fi
    if ((${#lane1[@]})); then
      RESULTS_DIR="$RESULTS_DIR" EVAL_CONCURRENCY=32 \
        bash "$ROOT/eval_checkpoint_lane.sh" 1 "$PORT1" "${lane1[@]}" \
        >>"$LOG_ROOT/grpo24-watch-gpu1.log" 2>&1 &
      pids+=("$!")
    fi
    for pid in "${pids[@]}"; do wait "$pid"; done
  fi

  sleep "$POLL_SECONDS"
done

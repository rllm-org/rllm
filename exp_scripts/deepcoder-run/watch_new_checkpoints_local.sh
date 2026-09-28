#!/usr/bin/env bash
# After the initial fixed evaluation queue drains, continuously enqueue newly
# completed .16 DeepCoder checkpoints on one local GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
GPU="${1:?GPU}"
PORT="${2:?PORT}"
INITIAL_PID="${3:?PID of the initial lane}"
MOD="${4:?step modulo 20 assigned to this lane (0 or 10)}"
[[ "$MOD" == 0 || "$MOD" == 10 ]]

WORK=/mnt/disk1t/deepcoder-prpo-checkpoint-eval
RESULTS="$WORK/results"
REMOTE_ROOT=/home/yanan/.deepcoder-checkpoints/deepcoder-prpo-b64-n1-16-c64-ni0-save10-ep3

echo "[watch gpu=$GPU] waiting for initial lane pid=$INITIAL_PID"
while kill -0 "$INITIAL_PID" 2>/dev/null; do sleep 60; done
echo "[watch gpu=$GPU] initial lane complete; polling .16"

remote_size_if_complete() {
  local step="$1"
  ssh -o ConnectTimeout=10 10.225.68.16 "
    d='$REMOTE_ROOT/global_step_${step}'
    test -f \"\$d/data.pt\" &&
    test -f \"\$d/actor/lora_train_meta.json\" &&
    test -f \"\$d/actor/model_world_size_2_rank_0.pt\" &&
    test -f \"\$d/actor/model_world_size_2_rank_1.pt\" &&
    test -f \"\$d/actor/optim_world_size_2_rank_0.pt\" &&
    test -f \"\$d/actor/optim_world_size_2_rank_1.pt\" &&
    find \"\$d\" -type f -printf '%s\\n' | awk '{s+=\$1} END {print s+0}'
  "
}

while true; do
  mapfile -t STEPS < <(
    ssh -o ConnectTimeout=10 10.225.68.16 \
      "find '$REMOTE_ROOT' -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\\n'" 2>/dev/null |
      sed -n 's/^global_step_\([0-9][0-9]*\)$/\1/p' | sort -n
  ) || true

  DID_WORK=0
  for STEP in "${STEPS[@]}"; do
    (( STEP > 170 )) || continue
    (( STEP % 20 == MOD )) || continue
    [[ ! -s "$RESULTS/host16_global_step_${STEP}/result.json" ]] || continue

    # A directory is visible while torch.save is still writing. Require the
    # complete rank/file set and an unchanged aggregate size across 30 s.
    SIZE1="$(remote_size_if_complete "$STEP" 2>/dev/null || true)"
    [[ "$SIZE1" =~ ^[0-9]+$ && "$SIZE1" -gt 1000000000 ]] || continue
    sleep 30
    SIZE2="$(remote_size_if_complete "$STEP" 2>/dev/null || true)"
    [[ "$SIZE2" == "$SIZE1" ]] || continue

    echo "[watch gpu=$GPU] enqueue global_step_$STEP bytes=$SIZE2"
    if bash "$ROOT/eval_checkpoint_lane.sh" "$GPU" "$PORT" "host16:$STEP"; then
      echo "[watch gpu=$GPU] global_step_$STEP complete"
    else
      echo "[watch gpu=$GPU] global_step_$STEP failed; will retry" >&2
      sleep 60
    fi
    DID_WORK=1
  done
  (( DID_WORK == 1 )) || sleep 60
done


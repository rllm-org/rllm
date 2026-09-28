#!/usr/bin/env bash
# Pre-stage SFT checkpoints to .29 while the GPUs are still busy with GRPO evals.
# Network/disk only. Publishes atomically; a partial copy never becomes $RAW.
set -u
W=/mnt/disk1t/deepcoder-prpo-checkpoint-eval
SRC=/home/yanan/.deepcoder-checkpoints/deepcoder-self-sft-current4360-r32-e4-len18000
for STEP in "$@"; do
  RAW="$W/raw16/sft16_global_step_${STEP}"
  if [[ -f "$RAW/actor/lora_train_meta.json" ]]; then echo "[$STEP] already staged"; continue; fi
  # Wait for the checkpoint to exist on .16 (step 544 lands when training ends).
  for ((i=0; i<240; i++)); do
    if ssh -o ConnectTimeout=10 10.225.68.16 "test -f $SRC/global_step_${STEP}/lora_train_meta.json" 2>/dev/null; then break; fi
    echo "[$STEP] not on .16 yet, waiting"; sleep 60
  done
  TMP="$RAW.incoming"; rm -rf -- "$TMP"; mkdir -p "$TMP/actor"
  echo "[$STEP] staging $(date '+%T')"
  rsync -a --partial --exclude 'optim_world_size_*' \
    "10.225.68.16:$SRC/global_step_${STEP}/" "$TMP/actor/" >/dev/null 2>&1
  if [[ -f "$TMP/actor/lora_train_meta.json" && -f "$TMP/actor/fsdp_config.json" ]] \
     && find "$TMP/actor" -maxdepth 1 -name 'model_world_size_*.pt' -print -quit | grep -q .; then
    mv "$TMP" "$RAW"; echo "[$STEP] published $(date '+%T') $(du -sh "$RAW" | cut -f1)"
  else
    echo "[$STEP] FAILED preflight, left at $TMP"
  fi
done
echo "ALL DONE $(date '+%F %T %Z')"
ls -1d "$W"/raw16/sft16_global_step_* 2>/dev/null

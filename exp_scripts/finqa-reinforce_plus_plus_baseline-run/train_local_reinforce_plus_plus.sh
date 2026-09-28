#!/usr/bin/env bash
# FinQA REINFORCE++-baseline comparison arm on a 2x48GB GPU host.
#
# This is deliberately a thin outer-layer wrapper around the completed GRPO
# launch recipe.  All model, LoRA, optimiser, policy-loss, clipping, KL,
# entropy, rollout, judge and sequence-length settings therefore remain owned
# by that recipe.  The only scientific change is the advantage estimator.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
GRPO_DIR="$(cd "$RUN_DIR/../finqa-grpo-run" && pwd)"
MODE="${1:-smoke}"
shift || true

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints"

# train_finqa_subset.py uses this seed before the rLLM dataloader is built.
# Zero is the completed GRPO arm's default and preserves its task subset/order
# contract.  The full formal split is not subsetted, but recording this still
# makes smoke selection reproducible.
export FINQA_SUBSET_SEED="${FINQA_SUBSET_SEED:-0}"

# env.sh still points judge telemetry at the pre-move GRPO location.  A shell
# wrapper cannot override it before train_16.sh sources env.sh, so pass a
# per-run path through the environment after loading the same credentials.
source "$GRPO_DIR/env.sh"
export FINQA_JUDGE_FINISH_LOG="$RUN_DIR/logs/judge_finish.tsv"
# The model snapshot is pre-cached on each training host.  Keep Transformers
# from performing an unrelated Hub metadata request during tokenizer loading;
# this also pins the formal run to the audited local revision.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# The two RTX 6000 Ada GPUs on .24 have a broken NCCL peer-to-peer path: the
# first cross-rank collective hangs with the default transport.  Host-local
# two-process probes have verified that disabling P2P makes the same
# collective complete immediately.  Keep this workaround scoped to this
# launcher rather than changing the shared GRPO recipe.
export NCCL_P2P_DISABLE=1

# Pass the cached snapshot itself to both Transformers and vLLM.  On .24 the
# executable model files are complete, but the Hub cache lacks three repository
# documentation files; recent huggingface_hub rejects that cache by model ID in
# offline mode even though the model is loadable.  A local path avoids both the
# metadata request and that irrelevant repository-completeness check.
LOCAL_MODEL_PATH="${FINQA_LOCAL_MODEL_PATH:-/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
if [[ ! -f "$LOCAL_MODEL_PATH/config.json" || ! -f "$LOCAL_MODEL_PATH/model.safetensors.index.json" ]]; then
  echo "complete local model snapshot not found: $LOCAL_MODEL_PATH" >&2
  exit 1
fi

COMMON_OVERRIDES=(
  algorithm.adv_estimator=reinforce_plus_plus_baseline
  rllm.algorithm.adv_estimator=reinforce_plus_plus_baseline
  actor_rollout_ref.rollout.n=8
  rllm.rollout.n=8
  rllm.algorithm.rollout_correction.tis_mode=null
  model.name="$LOCAL_MODEL_PATH"
  actor_rollout_ref.model.path="$LOCAL_MODEL_PATH"
  trainer.project_name=finqa-reinforce-plus-plus
  trainer.default_local_dir="$RUN_DIR/checkpoints/\${trainer.experiment_name}"
)

case "$MODE" in
  smoke)
    # Reuse the GRPO stability dataset path (640 randomly selected tasks) but
    # stop after two optimiser steps.  With batch=32 and n=8 this exercises
    # 64 distinct task prompts and 512 trajectories.
    exec "$GRPO_DIR/train_16.sh" stability \
      trainer.total_training_steps=2 \
      trainer.test_freq=-1 \
      trainer.save_freq=-1 \
      trainer.val_before_train=false \
      trainer.resume_mode=disable \
      trainer.experiment_name=qwen3-4b-local-rpp-n8-smoke \
      "${COMMON_OVERRIDES[@]}" \
      "$@"
    ;;
  formal)
    # Match the completed GRPO run: full 4030-task split, batch=32, n=8 and
    # ten epochs. 4030 // 32 = 125 complete batches per epoch; save once at
    # each exact epoch boundary and perform formal evaluation offline.
    exec "$GRPO_DIR/train_16.sh" epoch \
      trainer.total_epochs=10 \
      rllm.trainer.total_epochs=10 \
      trainer.total_training_steps=1250 \
      rllm.trainer.total_batches=1250 \
      trainer.test_freq=-1 \
      rllm.trainer.test_freq=-1 \
      trainer.save_freq=125 \
      rllm.trainer.save_freq=125 \
      trainer.val_before_train=false \
      rllm.trainer.val_before_train=false \
      trainer.resume_mode=auto \
      trainer.experiment_name=qwen3-4b-24-rpp-n8-formal \
      "${COMMON_OVERRIDES[@]}" \
      "$@"
    ;;
  *)
    echo "usage: $0 {smoke|formal} [hydra overrides...]" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
# Formal synchronous single-rollout PRPO run on the full FinQA train split.
# Keeps all implementation in the outer experiment layer and delegates to the
# smoke-validated launcher, which reuses rLLM/verl/vLLM unchanged.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"

# 4030 // 64 = 62 complete batches per epoch (62 examples are dropped).
# Fifty epochs therefore contain 3100 optimizer steps. Save once at each
# complete epoch boundary; formal evaluation is performed offline. Resuming from the archived run's last
# complete checkpoint (step 608) replays only the unsaved steps 609-620 before
# writing the exact epoch-10 checkpoint.
export FINQA_TRAIN_N=0
export FINQA_VAL_N=0
export FINQA_SUBSET_SEED="${FINQA_SUBSET_SEED:-20260908}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b-16-prpo-sync-b64-formal}"

exec "$RUN_DIR/train_16_prpo.sh" \
    data.train_batch_size=64 \
    rllm.data.train_batch_size=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    trainer.total_epochs=50 \
    rllm.trainer.total_epochs=50 \
    trainer.total_training_steps=3100 \
    rllm.trainer.total_batches=3100 \
    trainer.save_freq=62 \
    rllm.trainer.save_freq=62 \
    trainer.test_freq=-1 \
    rllm.trainer.test_freq=-1 \
    trainer.val_before_train=false \
    rllm.trainer.val_before_train=false \
    trainer.resume_mode=auto \
    "$@"

#!/bin/bash
set -x

# ============================================================================
# TINKER BACKEND TRAINING FOR VERIFIERS
# ============================================================================
#
# This uses Tinker for inference. You need:
# 1. A Tinker API key
# 2. Model available on Tinker
#
# Tinker handles:
#   - Model inference (remote)
#   - Weight management
#   - LoRA training
#
# ============================================================================

# ============================================================================
# CONFIGURE THESE FOR YOUR SETUP
# ============================================================================

# Load API key from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
    export TINKER_API_KEY
fi

# Tinker service URL (null uses default Tinker cloud)
TINKER_BASE_URL=null

# Model to train (HuggingFace path)
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"

# Verifiers environment configuration
VF_ENV_ID="primeintellect/alphabet-sort"

# LoRA configuration
LORA_RANK=32

# Training hyperparameters
GROUP_SIZE=16          # Rollouts per prompt (for GRPO)
LEARNING_RATE=2e-5
MAX_LENGTH=32768
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=32
TOTAL_EPOCHS=10

# Sampling parameters (MUST be 1.0 for Tinker workflow trainer)
TEMPERATURE=1.0
TOP_P=1.0

# Workflow configuration
N_PARALLEL_TASKS=256
RETRY_LIMIT=3

# Logging
PROJECT_NAME="verifiers-tinker"
EXPERIMENT_NAME="${VF_ENV_ID}-training"

# ============================================================================
# RUN TRAINING
# ============================================================================

python3 -m examples.verifiers_env.train \
    --config-name=tinker_rl_trainer \
    +backend=tinker \
    tinker_base_url=${TINKER_BASE_URL} \
    model.name="${MODEL_NAME}" \
    model.lora_rank=${LORA_RANK} \
    +verifiers.env_id="${VF_ENV_ID}" \
    algorithm.adv_estimator=grpo \
    training.group_size=${GROUP_SIZE} \
    training.learning_rate=${LEARNING_RATE} \
    training.max_length=${MAX_LENGTH} \
    sampling.temperature=${TEMPERATURE} \
    sampling.top_p=${TOP_P} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    workflow.n_parallel_tasks=${N_PARALLEL_TASKS} \
    workflow.retry_limit=${RETRY_LIMIT} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.test_freq=5 \
    trainer.save_freq=20 \
    trainer.val_before_train=true

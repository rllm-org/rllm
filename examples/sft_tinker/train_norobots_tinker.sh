#!/bin/bash

# Train on NoRobots dataset using Tinker SFT backend
# This replicates the tinker-cookbook's sl_basic.py setup
#
# Reference: tinker-cookbook/tinker_cookbook/recipes/sl_basic.py
# - Model: meta-llama/Llama-3.1-8B
# - Dataset: HuggingFaceH4/no_robots (loaded from DatasetRegistry)
# - Batch size: 128
# - Max length: 32768
# - Learning rate: 2e-4
# - LR schedule: linear
# - Train on: ALL_ASSISTANT_MESSAGES (cumulative method in rLLM)

set -e

# Check if TINKER_API_KEY is set
if [ -z "$TINKER_API_KEY" ]; then
    echo "Error: TINKER_API_KEY environment variable is not set"
    echo "Please set it with: export TINKER_API_KEY=your_api_key"
    exit 1
fi

# Configuration matching tinker-cookbook's sl_basic.py
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-linear}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LORA_RANK="${LORA_RANK:-32}"
EVAL_EVERY="${EVAL_EVERY:-8}"
SAVE_EVERY="${SAVE_EVERY:-20}"

# Renderer for Llama-3.1
RENDERER_NAME="${RENDERER_NAME:-llama3}"

# Training settings
PROJECT_NAME="${PROJECT_NAME:-rllm-tinker-norobots}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-sl_basic}"
LOG_DIR="${LOG_DIR:-/tmp/rllm-tinker-examples/sft_norobots}"

echo "=========================================="
echo "Training NoRobots with Tinker SFT Backend"
echo "=========================================="
echo "Replicating: tinker-cookbook/recipes/sl_basic.py"
echo "Using DatasetRegistry (similar to RL trainer)"
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: norobots (from DatasetRegistry)"
echo "  Batch size: $BATCH_SIZE"
echo "  Max length: $MAX_LENGTH"
echo "  Learning rate: $LEARNING_RATE"
echo "  LR schedule: $LR_SCHEDULE"
echo "  LoRA rank: $LORA_RANK"
echo "  Epochs: $NUM_EPOCHS"
echo "  Eval every: $EVAL_EVERY steps"
echo "  Save every: $SAVE_EVERY steps"
echo "  Renderer: $RENDERER_NAME"
echo "  Log dir: $LOG_DIR"
echo "=========================================="
echo ""

python train_norobots_tinker.py \
    model.name=$MODEL_NAME \
    model.lora_rank=$LORA_RANK \
    data.train_batch_size=$BATCH_SIZE \
    data.max_length=$MAX_LENGTH \
    data.renderer_name=$RENDERER_NAME \
    data.rllm.tokenize_and_mask_method=cumulative \
    optim.lr=$LEARNING_RATE \
    optim.lr_scheduler=$LR_SCHEDULE \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.save_freq=$SAVE_EVERY \
    trainer.test_freq=$EVAL_EVERY \
    trainer.default_local_dir=$LOG_DIR \
    "$@"


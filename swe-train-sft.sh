#!/bin/bash
# SWE SFT Training Script
# Usage: bash swe-train-sft.sh <model> [root]
#
# Examples:
#   # Train SFT from base model
#   bash swe-train-sft.sh Qwen3-8B
#
#   # Train with custom root directory
#   bash swe-train-sft.sh Qwen3-8B /mnt/bn/my-bucket

set -x

# ============ Arguments ============
MODEL_NAME=${1:?'Usage: bash swe-train-sft.sh <model_name> [root_dir]'}
ROOT_DIR=${2:-'/mnt/bn/trae-research-models/xujunjielong'}

# ============ Environment ============
export GLOO_SOCKET_IFNAME='eth0'
export NCCL_SOCKET_IFNAME='eth0'

export UV_INDEX_URL=https://bytedpypi.byted.org/simple/

export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

# ============ Config ============
WAND_PROJECT='xujunjielong'
EXPERIMENT_NAME='agentic-swe-sft'
LR=2e-5

# ============ Byted env ============
uv pip uninstall ray wandb bytedray byted-wandb
uv pip install bytedray[default,data,serve,bytedance] byted-wandb

# ============ Distributed training (compatible with 1~N nodes) ============
NNODES=${ARNOLD_WORKER_NUM:-1}
NPROC_PER_NODE=8

if [ "$NNODES" -gt 1 ]; then
    TORCHRUN_ARGS="--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --node_rank=${ARNOLD_ID:-0} --master_addr=${ARNOLD_WORKER_0_HOST} --master_port=${ARNOLD_WORKER_0_PORT}"
else
    TORCHRUN_ARGS="--standalone --nproc_per_node=$NPROC_PER_NODE"
fi

# ============ Run Training ============
uv run --no-sync torchrun \
    $TORCHRUN_ARGS \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/swe/R2EGym_SFT_Trajectories.parquet \
    data.val_files=data/swe/R2EGym_SFT_Trajectories.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=16384 \
    data.truncation=right \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=4 \
    optim.lr=$LR \
    trainer.total_epochs=2 \
    model.partial_pretrain=$ROOT_DIR/models/$MODEL_NAME \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$ROOT_DIR/experiments/verl/$EXPERIMENT_NAME \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="['console','wandb']" \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    2>&1 | tee $EXPERIMENT_NAME.log

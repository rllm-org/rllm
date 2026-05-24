#!/usr/bin/env bash
# Train the Sokoban agent with the tinker backend.

set -euo pipefail

cd "$(dirname "$0")"

python3 -u train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    training.group_size=8 \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=10 \
    rllm.trainer.project_name=sokoban \
    rllm.trainer.experiment_name=qwen3-4b-instruct \
    rllm.trainer.logger=[console,ui] \
    "$@"

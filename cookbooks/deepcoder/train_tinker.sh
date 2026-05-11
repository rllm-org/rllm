#!/usr/bin/env bash
# Train the deepcoder agent with the tinker (single-machine) backend.
#
# Prerequisites:
#   1. Install rllm with tinker extras:   uv pip install -e ".[tinker]"
#   2. Install this cookbook:              uv pip install --no-deps -e cookbooks/deepcoder
#   3. Prepare the dataset:                python cookbooks/deepcoder/prepare_deepcoder_data.py

set -euo pipefail

python -u train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    training.group_size=4 \
    data.train_batch_size=16 \
    data.val_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=20 \
    rllm.trainer.project_name=deepcoder \
    rllm.trainer.experiment_name=qwen3-4b-instruct \
    rllm.trainer.logger=[console,ui] \
    "$@"

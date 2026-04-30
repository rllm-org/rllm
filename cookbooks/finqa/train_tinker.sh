#!/usr/bin/env bash
# Train the finqa agent with the tinker (single-machine) backend.
#
# Prerequisites:
#   1. Install rllm with tinker extras:    uv pip install -e ".[tinker]"
#   2. Install this cookbook:               uv pip install --no-deps -e cookbooks/finqa
#   3. Prepare the dataset:                 python cookbooks/finqa/prepare_data.py
#   4. Set the judge key:                   OPENAI_API_KEY=…

set -euo pipefail

python -u train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-30B-A3B-Instruct-2507 \
    model.lora_rank=32 \
    training.group_size=8 \
    data.train_batch_size=16 \
    data.val_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=20 \
    rllm.trainer.project_name=finqa \
    rllm.trainer.experiment_name=qwen3-30b-instruct \
    rllm.trainer.logger=[console,ui] \
    "$@"

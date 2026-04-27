#!/usr/bin/env bash
# Train the math tool agent via train.py with Hydra overrides.
#
# Prerequisites:
#   1. Install rllm with tinker extras:  uv pip install -e ".[tinker]"
#   2. Install this cookbook:             uv pip install --no-deps -e cookbooks/math_tool_agent
#   3. Pull the datasets:                rllm dataset pull deepscaler_math && rllm dataset pull math500
#
# To enable UI logging, append: rllm.trainer.logger=[console,ui]

set -euo pipefail

python -u train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    training.group_size=8 \
    data.train_batch_size=32 \
    data.val_batch_size=500 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=5 \
    rllm.trainer.project_name=math_tool_agent \
    rllm.trainer.experiment_name=qwen3-4b-instruct \
    rllm.trainer.logger=[console,ui] \
    "$@"

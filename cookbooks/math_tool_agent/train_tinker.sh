#!/usr/bin/env bash
# Train the math tool agent via train.py with Hydra overrides.
#
# Prerequisites:
#   1. Install rllm with tinker extras:  uv pip install -e ".[tinker]"
#   2. Install this cookbook:             uv pip install --no-deps -e cookbooks/math_tool_agent
#   3. Pull the dataset:                 rllm dataset pull gsm8k
#
# To enable UI logging, append: rllm.trainer.logger=[console,ui]

set -euo pipefail

python -u train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    training.group_size=8 \
    rllm.workflow.n_parallel_tasks=32 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=5 \
    rllm.trainer.project_name=math_tool_agent \
    rllm.trainer.experiment_name=qwen3-4b-instruct \
    rllm.trainer.logger=[console,ui] \
    "$@"

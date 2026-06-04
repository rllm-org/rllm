#!/usr/bin/env bash
# Train solver-judge via train.py with Hydra overrides.
#
# Prerequisites:
#   1. Install rllm with tinker extras:  uv pip install -e ".[tinker]"
#   2. Install this cookbook:             uv pip install --no-deps -e cookbooks/solver_judge_flow
#   3. Pull the dataset:                  rllm dataset pull countdown
#
# To enable UI logging, append: rllm.trainer.logger=[console,ui]

set -euo pipefail

python -u train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    training.group_size=4 \
    data.train_batch_size=32 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=10 \
    rllm.trainer.project_name=solver_judge \
    rllm.trainer.experiment_name=qwen3-4b-instruct \
    rllm.trainer.logger=[console,ui] \
    "$@"

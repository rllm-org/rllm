#!/usr/bin/env bash
# Train geo3k VLM geometry solver via train.py with Hydra overrides.
#
# Prerequisites:
#   1. Install rllm with tinker extras:  uv pip install -e ".[tinker]"
#   2. Install this cookbook:             uv pip install -e cookbooks/geo3k
#   3. Pull the dataset:                 rllm dataset pull geo3k

set -euo pipefail

python -u cookbooks/geo3k/train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-VL-30B-A3B-Instruct \
    model.lora_rank=32 \
    training.group_size=8 \
    rllm.trainer.total_epochs=3 \
    rllm.trainer.test_freq=10 \
    rllm.trainer.project_name=geo3k \
    rllm.trainer.experiment_name=qwen3-vl-30b-instruct \
    "$@"

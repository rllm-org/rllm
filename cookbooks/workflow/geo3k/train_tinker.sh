#!/usr/bin/env bash
# Train Geo3KWorkflow (legacy Workflow API) via the unified trainer + tinker backend.
#
# Prerequisites:
#   1. Install rllm with tinker extras:  uv pip install -e ".[tinker]"
#   2. Pull the dataset:                  rllm dataset pull geo3k
#   3. Run from this cookbook dir:        cd cookbooks/workflow/geo3k

set -euo pipefail

python -u train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-VL-30B-A3B-Instruct \
    model.lora_rank=32 \
    training.group_size=8 \
    rllm.rollout.train.temperature=0.6 \
    rllm.rollout.val.temperature=0.6 \
    rllm.trainer.total_epochs=3 \
    rllm.trainer.test_freq=10 \
    rllm.trainer.project_name=geo3k_workflow \
    rllm.trainer.experiment_name=qwen3-vl-30b-instruct \
    "$@"

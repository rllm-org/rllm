#!/usr/bin/env bash
# Train Geo3KWorkflow (legacy Workflow API) via the unified trainer + fireworks backend.
#
# Prerequisites:
#   1. Install rllm with fireworks extras:  uv pip install -e ".[fireworks]"
#   2. Pull the dataset:                     rllm dataset pull geo3k
#   3. Run from this cookbook dir:           cd cookbooks/workflow/geo3k
#   4. Set your API key:                     export FIREWORKS_API_KEY=...
#
# The trainer job and inference deployment are provisioned on Fireworks at
# startup and torn down on shutdown.
#
# WARNING: geo3k is a vision task. The Fireworks training shapes catalog
# currently lists RFT as unsupported for Qwen3-VL models (only an SFT LoRA
# shape exists). Verify VL RFT support in the catalog before running:
# https://docs.fireworks.ai/fine-tuning/training-api/training-shapes

set -euo pipefail

python -u train.py \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/qwen3-vl-8b-instruct \
    model.tokenizer_model=Qwen/Qwen3-VL-8B-Instruct \
    model.lora_rank=32 \
    fireworks_config.policy_trainer_shape_id=accounts/fireworks/trainingShapes/qwen3-vl-8b-256k-h200-lora \
    training.group_size=8 \
    rllm.trainer.total_epochs=3 \
    rllm.trainer.test_freq=10 \
    rllm.trainer.project_name=geo3k_workflow \
    rllm.trainer.experiment_name=qwen3-vl-8b-instruct-fireworks \
    rllm.trainer.logger=[console,ui] \
    "$@"

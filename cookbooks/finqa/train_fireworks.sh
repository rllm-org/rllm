#!/usr/bin/env bash
# Train the finqa agent with the fireworks (managed) backend.
#
# Prerequisites:
#   1. Install rllm with fireworks extras:  uv pip install -e ".[fireworks]"
#   2. Install this cookbook:                uv pip install --no-deps -e cookbooks/finqa
#   3. Prepare the dataset:                  python cookbooks/finqa/prepare_finqa_data.py
#   4. Set the judge key:                    OPENAI_API_KEY=…
#   5. Set your API key:                     export FIREWORKS_API_KEY=...
#
# The trainer job and inference deployment are provisioned on Fireworks at
# startup and torn down on shutdown.
#
# NOTE: The tinker variant uses Qwen3-30B-A3B-Instruct-2507, which has no
# RFT LoRA training shape on Fireworks. We use Qwen3.5-35B-A3B (the closest
# MoE model with RFT LoRA support) instead. See the training shapes catalog:
# https://docs.fireworks.ai/fine-tuning/training-api/training-shapes

set -euo pipefail

python -u train.py \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/qwen3p5-35b-a3b \
    model.tokenizer_model=Qwen/Qwen3.5-35B-A3B \
    model.lora_rank=32 \
    fireworks_config.policy_trainer_shape_id=accounts/fireworks/trainingShapes/qwen3p5-35b-a3b-256k-lora \
    training.group_size=8 \
    data.train_batch_size=16 \
    data.val_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=20 \
    rllm.trainer.project_name=finqa \
    rllm.trainer.experiment_name=qwen3p5-35b-a3b-fireworks \
    rllm.trainer.logger=[console,ui] \
    "$@"

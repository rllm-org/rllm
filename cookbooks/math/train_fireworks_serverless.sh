#!/usr/bin/env bash
# Train the math agent with pooled Fireworks serverless training.
#
# Prerequisites:
#   1. Install rllm with fireworks extras:  uv pip install -e ".[fireworks]"
#   2. Install this cookbook:                uv pip install --no-deps -e cookbooks/math
#   3. Pull the datasets:                    rllm dataset pull hendrycks_math && rllm dataset pull math500
#   4. Set your API key:                     export FIREWORKS_API_KEY=...
#
# No trainer job or inference deployment is provisioned. The shared serverless
# session supplies both the LoRA training client and snapshot sampling clients.

set -euo pipefail

python -u train.py \
    rllm/backend=fireworks_serverless \
    model.name=accounts/fireworks/models/qwen3p5-9b \
    model.tokenizer_model=Qwen/Qwen3.5-9B \
    model.lora_rank=32 \
    training.max_length=16384 \
    training.group_size=8 \
    data.train_batch_size=32 \
    data.val_batch_size=500 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=10 \
    rllm.trainer.project_name=math \
    rllm.trainer.experiment_name=qwen3p5-9b-fireworks-serverless \
    rllm.trainer.logger=[console,ui] \
    "$@"

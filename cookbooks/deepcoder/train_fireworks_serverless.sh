#!/usr/bin/env bash
# Train the DeepCoder agent with pooled Fireworks serverless training.

set -euo pipefail

python -u train.py \
    rllm/backend=fireworks_serverless \
    model.name=accounts/fireworks/models/qwen3p5-9b \
    model.tokenizer_model=Qwen/Qwen3.5-9B \
    model.lora_rank=32 \
    training.max_length=32768 \
    training.group_size=4 \
    data.train_batch_size=16 \
    data.val_batch_size=50 \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=20 \
    rllm.trainer.project_name=deepcoder \
    rllm.trainer.experiment_name=qwen3p5-9b-fireworks-serverless \
    rllm.trainer.logger=[console,ui] \
    "$@"

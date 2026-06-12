#!/usr/bin/env bash
# Train the math tool agent via train.py with Hydra overrides (fireworks backend).
#
# Prerequisites:
#   1. Install rllm with fireworks extras:  uv pip install -e ".[fireworks]"
#   2. Install this cookbook:                uv pip install --no-deps -e cookbooks/math_tool_agent
#   3. Pull the datasets:                    rllm dataset pull deepscaler_math && rllm dataset pull math500
#   4. Set your API key:                     export FIREWORKS_API_KEY=...
#
# The trainer job and inference deployment are provisioned on Fireworks at
# startup and torn down on shutdown.
#
# To enable UI logging, append: rllm.trainer.logger=[console,ui]

set -euo pipefail

python -u train.py \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/qwen3-4b-instruct-2507 \
    model.tokenizer_model=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    fireworks_config.policy_trainer_shape_id=accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora \
    training.group_size=8 \
    data.train_batch_size=32 \
    data.val_batch_size=500 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=10 \
    rllm.trainer.project_name=math_tool_agent \
    rllm.trainer.experiment_name=qwen3-4b-instruct-fireworks \
    rllm.trainer.logger=[console,ui] \
    "$@"

#!/usr/bin/env bash
# Quick-start script for SDK + Unified Trainer + Tinker backend.
#
# Prerequisites:
#   1. Install rllm with tinker backend:  uv pip install -e ".[tinker]"
#   2. Prepare dataset:  python -m examples.countdown.prepare_countdown_data
#
# Usage:
#   bash examples/sdk/unified_tinker/run.sh          # use defaults
#   bash examples/sdk/unified_tinker/run.sh model.name=Qwen/Qwen3-1.7B  # override

set -euo pipefail

python -m examples.sdk.unified_tinker.train_countdown_sdk_tinker \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-8B \
    model.lora_rank=32 \
    training.group_size=8 \
    validation.group_size=1 \
    training.learning_rate=2e-5 \
    sampling.train.temperature=1.0 \
    sampling.train.top_p=1.0 \
    rllm.algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=5 \
    rllm.trainer.val_before_train=false \
    rllm.trainer.project_name=sdk-unified-tinker \
    rllm.trainer.experiment_name=countdown \
    rllm.workflow.n_parallel_tasks=64 \
    rllm.sdk.proxy.port=8089 \
    "$@"

#!/usr/bin/env bash
# One-step Countdown smoke run on Fireworks serverless training.

set -euo pipefail

python -m examples.countdown.unified_trainer.train_countdown_unified_fireworks \
    rllm/backend=fireworks_serverless \
    model.name=accounts/fireworks/models/qwen3p5-9b \
    model.tokenizer_model=Qwen/Qwen3.5-9B \
    model.lora_rank=32 \
    rollout_engine.renderer_name=qwen3_5 \
    rollout_engine.reasoning_effort=none \
    training.group_size=4 \
    training.learning_rate=2e-5 \
    training.max_length=4096 \
    validation.group_size=1 \
    rllm.workflow.n_parallel_tasks=8 \
    rllm.workflow.retry_limit=1 \
    rllm.workflow.raise_on_error=false \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.train_batch_size=4 \
    data.val_batch_size=16 \
    rllm.data.max_prompt_length=2048 \
    rllm.data.max_response_length=1024 \
    rllm.data.train_batch_size=4 \
    rllm.data.val_batch_size=16 \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.total_batches=1 \
    rllm.trainer.logger='[console]' \
    rllm.trainer.project_name='rllm-countdown' \
    rllm.trainer.experiment_name='countdown-qwen3p5-9b-fireworks-serverless' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=0 \
    rllm.trainer.save_freq=1 \
    "$@"

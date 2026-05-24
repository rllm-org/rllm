#!/usr/bin/env bash
# Train the step-based (non-cumulative) Sokoban agent with the verl backend.
# Uses stepwise GRPO with advantage broadcasting — matching verl-agent's approach.

set -euo pipefail

cd "$(dirname "$0")"

python3 -u train_step.py \
    rllm/backend=verl \
    model.name=Qwen/Qwen3-4B \
    model.lora_rank=32 \
    training.group_size=24 \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.stepwise_advantage.enable=True \
    rllm.stepwise_advantage.mode=broadcast \
    rllm.disable_thinking=True \
    rllm.trainer.total_epochs=300 \
    rllm.trainer.test_freq=5 \
    rllm.trainer.save_freq=10 \
    rllm.trainer.project_name=sokoban \
    rllm.trainer.experiment_name=qwen3-4b-step-grpo \
    rllm.trainer.logger=[console,wandb] \
    rllm.rollout.n=24 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.rollout.val.temperature=0.7 \
    rllm.rollout.val.top_p=0.8 \
    "$@"

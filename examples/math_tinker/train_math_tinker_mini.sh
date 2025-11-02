#!/bin/bash
# Train a math agent using TinkerAgentTrainer (simplified wrapper)
# This exactly matches train_seperate.sh configuration but uses TinkerAgentTrainer

set -x

# Model to use (matching math_rl recipe default for Hendrycks MATH)
MODEL_PATH=Qwen/Qwen3-8B

# IMPORTANT: First prepare the dataset with proper filtering
# python -m examples.simple_math_tinker.prepare_tinker_math_dataset

# Run training with EXACT math_rl-matching configuration:
# Key parameters (matching original):
# 1. group_size=16 - matches math_rl GRPO group size for MATH
# 2. batch_size=128 - matches math_rl groups_per_batch
# 3. norm_adv_by_std_in_grpo=false - math_rl doesn't normalize by std
# 4. max_tokens=512 - matches math_rl default
# 5. Uses MathAgentWithFewshot for few-shot prompting

python3 -m examples.math_tinker.train_math_tinker \
    tinker.model.name=$MODEL_PATH \
    tinker.model.lora_rank=32 \
    tinker.training.group_size=16 \
    tinker.training.learning_rate=4e-5 \
    tinker.training.save_every=20 \
    tinker.sampling.temperature=1.0 \
    tinker.sampling.top_p=1.0 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.train_batch_size=4 \
    data.val_batch_size=32 \
    agent.max_steps=1 \
    trainer.total_epochs=1 \
    trainer.logger=['wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='rllm-tinker-math' \
    trainer.val_before_train=False \
    trainer.test_freq=20 \
    trainer.save_freq=20 \
    trainer.default_local_dir='/tmp/rllm-tinker-math'




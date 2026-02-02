#!/bin/bash
# Train a math agent using TinkerAgentTrainer (simplified wrapper)
# This exactly matches train_seperate.sh configuration but uses TinkerAgentTrainer
#
# To resume from a Tinker model ID (from Tinker console):
#   trainer.resume_from_tinker_id='tinker://<uuid>/weights/<checkpoint_name>'
#   where <checkpoint_name> is the name of the checkpoint to resume from
#   e.g. trainer.resume_from_tinker_id='tinker://7af7f6f0-e8f9-4124-ab93-e7f08eb54a9d/weights/000060'

set -x

# Model to use (matching math_rl recipe default for Hendrycks MATH)
MODEL_PATH=Qwen/Qwen3-30B-A3B

# To match the performance of tinker's original rl_loop.py
# Apply a small patch at rllm/engine/agent_execution_engine.py
# Comment out the line that sets reward to 0.0:

# if response_token_len - len(env_msg_tokens) > self.max_response_length:
#     cur_step.reward = 0.0


python -m examples.math_tinker.train_math_tinker \
    model.name=$MODEL_PATH \
    model.lora_rank=32 \
    training.group_size=16 \
    training.learning_rate=4e-5 \
    sampling.temperature=1.0 \
    sampling.top_p=1.0 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.train_batch_size=128 \
    data.val_batch_size=500 \
    agent.max_steps=1 \
    trainer.total_epochs=1 \
    trainer.logger=['wandb'] \
    trainer.project_name='rllm-tinker' \
    trainer.experiment_name='rllm-tinker-math' \
    trainer.val_before_train=False \
    trainer.test_freq=20 \
    trainer.save_freq=20 \
    trainer.default_local_dir='/tmp/rllm-tinker-math'
#!/bin/bash
set -x

# Environment variables for vLLM (managed internally by verl)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# ============================================================================
# CONFIGURE THESE FOR YOUR SETUP
# ============================================================================

# Model to train (HuggingFace path or local path)
MODEL_PATH="Qwen/Qwen3-4B"

# Verifiers environment configuration
VF_ENV_ID="math"  # The verifiers environment to use
# VF_ENV_ARGS can be passed as JSON, e.g., '{"difficulty": "hard"}'

# GPU configuration
N_GPUS=1  # Number of GPUs per node
NNODES=1  # Number of nodes

# Training hyperparameters
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=128
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=2048
TOTAL_EPOCHS=100
MAX_STEPS=10  # Max turns for multi-turn environments

# Sampling parameters
ROLLOUT_N=8  # Number of samples per prompt (for GRPO/PPO)
TEMPERATURE=0.7

# Logging
PROJECT_NAME="verifiers-rl"
EXPERIMENT_NAME="${VF_ENV_ID}-training"

# ============================================================================
# RUN TRAINING
# ============================================================================

python3 -m examples.verifiers_env.train \
    verifiers.env_id="${VF_ENV_ID}" \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    rllm.workflow.use_workflow=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    rllm.agent.max_steps=${MAX_STEPS} \
    trainer.total_epochs=${TOTAL_EPOCHS}

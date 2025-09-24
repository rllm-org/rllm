#!/bin/bash
set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct" 

# GPU selection (using GPU 4 here)
export CUDA_VISIBLE_DEVICES=4

# GPU configuration
NNODES=1
GPUS_PER_NODE=1 

# Parallelism settings
TP=1  # Tensor parallelism 
PP=1  # Pipeline parallelism 
EP=1  # Expert Parallel (EP > 1 for MoE models)

# Run DeepScaler training with Megatron
python -m examples.deepscaler.train_deepscaler_megatron \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP \
    critic.megatron.tensor_model_parallel_size=$TP \
    critic.megatron.pipeline_model_parallel_size=$PP \
    critic.megatron.expert_model_parallel_size=$EP
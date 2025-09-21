#!/bin/bash
# Megatron training script for DeepScaler with DatasetRegistry
set -x

# vLLM environment variables for optimal performance
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # Changed to True to avoid memory fragmentation
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# Additional memory optimization settings
export CUDA_LAUNCH_BLOCKING=0  # Allow async CUDA operations
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"  # Optimize for specific GPU architectures

# Model configuration
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # DeepScaler model

# GPU selection (optional - comment out to use all GPUs)
# export CUDA_VISIBLE_DEVICES=4,5  # Use only GPU 4 and 5
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use first 4 GPUs
# export CUDA_VISIBLE_DEVICES=4,5,6,7  # Use last 4 GPUs

# GPU configuration
NNODES=1
GPUS_PER_NODE=2  # Adjust based on CUDA_VISIBLE_DEVICES if set

# Parallelism settings
TP=2  # Tensor Parallel
PP=1  # Pipeline Parallel
EP=1  # Expert Parallel (increase for MoE models)

# Run DeepScaler training with Megatron
# Override strategy parameters directly instead of using config-name
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5} \
python -m examples.deepscaler.train_deepscaler_megatron \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.ref.strategy=megatron \
    critic.strategy=megatron \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP \
    critic.megatron.tensor_model_parallel_size=$TP \
    critic.megatron.pipeline_model_parallel_size=$PP \
    critic.megatron.expert_model_parallel_size=$EP
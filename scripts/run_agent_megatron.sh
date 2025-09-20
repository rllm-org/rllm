#!/bin/bash
# Megatron training script for DeepScaler with DatasetRegistry

# Model configuration
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  # DeepScaler model

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
# Uses DatasetRegistry - no need to specify data paths
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5} \
python -m rllm.examples.deepscaler.train_deepscaler_megatron \
    --config-name agent_ppo_trainer_megatron \
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
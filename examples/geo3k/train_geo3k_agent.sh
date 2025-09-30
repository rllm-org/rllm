#!/bin/bash
# GEO3K Multimodal Agent Training Script
# This script follows RLLM patterns while enabling multimodal training through Verl

set -x

# Environment setup for multimodal training
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export TOKENIZERS_PARALLELISM=false

# Use SGLang for multimodal support (following Verl's GEO3K example)
# Uncomment these for vLLM if preferred
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export VLLM_USE_V1=1
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# Find the directory where rllm package is located
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

# Train GEO3K agent with multimodal support
# Configuration based on Verl's geo3k_multiturn examples and RLLM patterns
python3 -m examples.geo3k.train_geo3k_agent \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=True \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    critic.ppo_micro_batch_size_per_gpu=2 \
    rllm.agent.name=geo3k_agent \
    rllm.agent.max_steps=1 \
    rllm.env.name=math \
    rllm.mask_truncated_samples=False \
    rllm.stepwise_advantage.enable=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-geo3k-multimodal' \
    trainer.experiment_name='geo3k-multimodal-qwen2vl-2b' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 \
    trainer.default_hdfs_dir=null
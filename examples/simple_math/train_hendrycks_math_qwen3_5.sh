#!/usr/bin/env bash
# Qwen3.5 GRPO on Hendrycks MATH. Variant of train_hendrycks_math.sh.
#
# Qwen3.5 requires: actor.strategy=fsdp2 (megatron not yet supported),
# Qwen3_5DecoderLayer wrap policy, vLLM gdn_prefill_backend=triton, and
# trust_remote_code=True.
#
# Defaults are tuned for 8x141GB H200; for 80GB GPUs set OFFLOAD=1
# MICRO_BS=1.
#
# Usage:
#   python3 examples/simple_math/prepare_math_dataset.py
#   bash examples/simple_math/train_hendrycks_math_qwen3_5.sh

set -ex

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_GDN_PREFILL_BACKEND=triton

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B}
N_GPUS=${N_GPUS:-8}
N_NODES=${N_NODES:-1}
TP_SIZE=${TP_SIZE:-2}
MICRO_BS=${MICRO_BS:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
OFFLOAD=${OFFLOAD:-0}

if [[ "${OFFLOAD}" == "1" ]]; then
    OFFLOAD_FLAGS=(
        ++actor_rollout_ref.actor.fsdp_config.offload_policy=True
        ++actor_rollout_ref.ref.fsdp_config.offload_policy=True
        actor_rollout_ref.actor.fsdp_config.param_offload=True
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
        actor_rollout_ref.ref.fsdp_config.param_offload=True
    )
else
    OFFLOAD_FLAGS=(
        ++actor_rollout_ref.actor.fsdp_config.offload_policy=False
        ++actor_rollout_ref.ref.fsdp_config.offload_policy=False
        actor_rollout_ref.actor.fsdp_config.param_offload=False
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
        actor_rollout_ref.ref.fsdp_config.param_offload=False
    )
fi

python3 -m examples.simple_math.train_hendrycks_math \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.return_raw_chat=True \
    data.truncation=error \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    ++actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BS} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    ++actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer] \
    ++actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer] \
    ++actor_rollout_ref.actor.fsdp_config.fsdp_size=${N_GPUS} \
    ++actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    ++actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    "${OFFLOAD_FLAGS[@]}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
    ++actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    rllm.mask_truncated_samples=False \
    rllm.agent.max_steps=1 \
    rllm.stepwise_advantage.enable=False \
    rllm.workflow.use_workflow=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name='rllm-qwen3_5' \
    trainer.experiment_name='hendrycks-math-qwen3_5-4b' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${N_NODES} \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=8

#!/bin/bash

set -x

export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Use both GPUs (0 and 1)
export CUDA_VISIBLE_DEVICES=0,1

python3 -m rllm.experimental.test.test_skyrl_simple_math \
    rllm/backend=skyrl \
    trainer.policy.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.placement.policy_num_gpus_per_node=1 \
    trainer.placement.ref_num_gpus_per_node=1 \
    trainer.placement.colocate_all=false \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.total_batches=100 \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=5 \
    rllm.trainer.experiment_name=skyrl_simple_math_test \
    rllm.rejection_sample.min_trajs_per_group=1 \
    trainer.policy.optimizer_config.lr=1e-5 \
    trainer.critic.optimizer_config.lr=5e-5 \
    +data.train_batch_size=32 \
    +data.val_batch_size=16 \
    generator.n_samples_per_prompt=8 \
    generator.num_inference_engines=1 \
    generator.inference_engine_tensor_parallel_size=1 \
    generator.gpu_memory_utilization=0.7 \
    generator.batched=false \
    generator.async_engine=true \
    generator.max_num_seqs=255 \
    +ray_init.num_cpus=32 

#!/bin/bash

set -x

export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Override this externally if needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

python3 -m rllm.experimental.test.test_skyrl_solver_judge \
    rllm/backend=skyrl \
    trainer.policy.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    trainer.placement.policy_num_gpus_per_node=1 \
    trainer.placement.ref_num_gpus_per_node=1 \
    trainer.placement.colocate_all=false \
    trainer.epochs=1 \
    rllm.trainer.total_batches=100 \
    trainer.eval_before_train=false \
    trainer.eval_interval=10 \
    trainer.val_max_samples=8 \
    trainer.ckpt_interval=20 \
    trainer.run_name=skyrl_solver_judge_test \
    rllm.episode_logging.log_episodes=true \
    rllm.rejection_sample.min_trajs_per_group=1 \
    trainer.policy.optimizer_config.lr=1e-6 \
    trainer.critic.optimizer_config.lr=5e-5 \
    trainer.algorithm.use_kl_loss=false \
    trainer.algorithm.eps_clip_high=0.28 \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    generator.n_samples_per_prompt=4 \
    generator.eval_n_samples_per_prompt=1 \
    generator.num_inference_engines=1 \
    generator.inference_engine_tensor_parallel_size=1 \
    generator.gpu_memory_utilization=0.7 \
    generator.batched=false \
    generator.async_engine=true \
    generator.max_num_seqs=255 \
    +ray_init.num_cpus=32

set -x

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

python -m examples.geo3k.unified_trainer.train_geo3k_unified_verl \
    rllm/backend=verl \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-2B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.adv_estimator=grpo \
    rllm.agent.max_steps=1 \
    rllm.rollout.n=8 \
    rllm.rollout.n_val=1 \
    rllm.rollout.sampling.train.temperature=0.6 \
    rllm.rollout.sampling.val.temperature=0.6 \
    rllm.rollout.sampling.val.top_p=0.9 \
    rllm.algorithm.eps_clip_high=0.28 \
    rllm.stepwise_advantage.enable=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-geo3k' \
    trainer.experiment_name='geo3k-unified-verl-qwen3-vl-2b' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=10 \
    trainer.total_epochs=3

pkill -9 -f 'ray::WorkerDict'

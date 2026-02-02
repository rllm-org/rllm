set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

echo "Store path: ${HOME}/rllm-traces.db"

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

python3 -m examples.sdk.simple_math.train_hendrycks_math \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='simple-math-simple-workflow' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=10 \
    rllm.agent.max_steps=1 \
    rllm.stepwise_advantage.enable=False \
    rllm.workflow.use_workflow=True \
    trainer.total_epochs=100 \
    rllm.sdk.proxy.host=127.0.0.1 \
    rllm.sdk.proxy.port=4000 \
    rllm.sdk.proxy.mode=subprocess \
    rllm.sdk.store.path="${HOME}/rllm-traces.db"

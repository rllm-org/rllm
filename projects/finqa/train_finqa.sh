set -x

unset ROCR_VISIBLE_DEVICES 
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export TOKENIZERS_PARALLELISM=true

export ACTOR_BUDGET=16384
export REF_BUDGET=40960

# --- FAST I/O SETUP ---
# Copy 7k+ small JSON files from slow NFS to fast local NVMe (/scratch)
if [ -d "/scratch" ] && [ -d "data/company_tables" ]; then
    echo "Copying tables to /scratch for fast CPU access..."
    LOCAL_DIR="/scratch/finqa_${SLURM_JOB_ID:-$USER}"
    mkdir -p "$LOCAL_DIR"
    cp -r data/company_tables "$LOCAL_DIR/"
    export FINQA_TABLES_ROOT="$LOCAL_DIR/company_tables"
fi
# ----------------------

python3 -m src.train_finqa_with_tool \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.gen_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=12288 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ACTOR_BUDGET \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=48000 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$REF_BUDGET \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$REF_BUDGET \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.entropy_coeff=0.002 \
    algorithm.kl_ctrl.kl_coef=0.01 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='4b-finqa-tool-setting-eleven' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=checkpoints/rllm-agent/4b-finqa-tool-setting-eleven/global_step_100 \
    rllm.agent.max_steps=20 \
    rllm.stepwise_advantage.enable=False \
    rllm.rejection_sample.enable=False \
    rllm.workflow.n_parallel_tasks=2048 \
    trainer.total_epochs=10

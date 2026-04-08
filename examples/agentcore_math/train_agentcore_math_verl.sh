#!/usr/bin/env bash
# Requires megatron deps: bash scripts/install_megatron.sh <cu128|cu129|cu130|...>
set -eux

# Load environment variables (AGENTCORE_AGENT_ARN, AGENTCORE_S3_BUCKET)
set -a && source .env && set +a

export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# Clean up stale ZMQ sockets from previous crashed runs
rm -f /tmp/rl-colocate-zmq-*.sock

MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507

python -m examples.agentcore_math.train_agentcore_math_verl \
    rllm/backend=verl \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    rllm.algorithm.use_rllm=true \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length=14336 \
    data.max_response_length=2048 \
    +model.name=$MODEL_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.lora.rank=16 \
    +actor_rollout_ref.model.lora.alpha=16 \
    +actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel=false \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.sequence_parallel=false \
    trainer.logger="['console','ui']" \
    trainer.project_name=agentcore-math \
    trainer.experiment_name=gsm8k-agentcore-verl-megatron-4b-instruct \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=disable \
    rllm.remote_runtime.enabled=true \
    rllm.remote_runtime.backend=agentcore \
    rllm.remote_runtime.backend_config.agent_runtime_arn=$AGENTCORE_AGENT_ARN \
    rllm.remote_runtime.backend_config.s3_bucket=$AGENTCORE_S3_BUCKET \
    rllm.remote_runtime.backend_config.tps_limit=25 \
    rllm.remote_runtime.session_timeout=300

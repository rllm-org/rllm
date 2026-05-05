#!/usr/bin/env bash
# Train solver-judge with the verl (distributed GPU) backend.
#
# Prerequisites:
#   1. Install rllm with verl extras:     uv pip install -e ".[verl]"
#   2. Install this cookbook:              uv pip install --no-deps -e cookbooks/solver_judge_flow
#   3. Pull the dataset:                  rllm dataset pull countdown

set -euo pipefail

unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507

python -u train.py \
    rllm/backend=verl \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    rllm.algorithm.use_rllm=true \
    data.train_batch_size=32 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    +model.name=$MODEL_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.lora.rank=32 \
    +actor_rollout_ref.model.lora.alpha=32 \
    +actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    +actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.logger="['console','ui']" \
    trainer.project_name=solver_judge \
    trainer.experiment_name=qwen3-4b-instruct-verl \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=disable \
    "$@"

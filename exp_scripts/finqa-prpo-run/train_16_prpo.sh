#!/usr/bin/env bash
# Synchronous single-rollout PRPO smoke for FinQA on .16.
#
# This is intentionally an outer-layer launcher. It reuses the existing FinQA
# driver and the unmodified rLLM/verl/vLLM training stack. It does not enable
# fully asynchronous training or stale-rollout correction.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
GRPO_DIR="$(cd "$RUN_DIR/../finqa-grpo-run" && pwd)"

source "$GRPO_DIR/env.sh"
source "$VENV/bin/activate"

# env.sh predates the move into gitlab/tail/rllm and contains the former
# absolute log path. Keep this experiment's judge telemetry self-contained.
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints"
export FINQA_JUDGE_FINISH_LOG="$RUN_DIR/logs/judge_finish.tsv"

# Use a reproducible, diverse pool. Dataset.shuffle(seed=...) is performed by
# train_finqa_subset.py before the rLLM dataloader builds batches.
export FINQA_SUBSET_SEED="${FINQA_SUBSET_SEED:-20260908}"
export FINQA_TRAIN_N="${FINQA_TRAIN_N:-640}"
export FINQA_VAL_N=0

cd /home/yanan/agents/rllm/cookbooks/finqa

MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b-16-prpo-sync-smoke}"

exec python -u "$GRPO_DIR/train_finqa_subset.py" \
    rllm/backend=verl \
    algorithm.adv_estimator=prpo \
    rllm.algorithm.adv_estimator=prpo \
    rllm.algorithm.rollout_correction.tis_mode=null \
    rllm.async_training.enable=false \
    rllm.rejection_sample.min_trajs_per_group=1 \
    rllm.algorithm.use_rllm=true \
    +model.name="$MODEL_PATH" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.002 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.logger="['console']" \
    trainer.project_name=finqa-prpo \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.val_before_train=false \
    trainer.resume_mode=disable \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$RUN_DIR/checkpoints/\${trainer.experiment_name}" \
    data.train_batch_size=32 \
    rllm.data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.n=1 \
    rllm.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.42 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    "$@"

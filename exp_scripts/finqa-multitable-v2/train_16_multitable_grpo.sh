#!/usr/bin/env bash
# FinQA multi-table-v2 GRPO on 2x RTX 6000 Ada (.16/.24).
# Experiment-layer wrapper only; rLLM/veRL/vLLM remain untouched.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
GRPO_DIR="$(cd "$RUN_DIR/../finqa-grpo-run" && pwd)"
RLLM_DIR=/home/yanan/agents/rllm
MODE="${1:-smoke}"
shift || true

source "$GRPO_DIR/env.sh"
source "$VENV/bin/activate"
export RLLM_HOME="$GRPO_DIR/.rllm_multilane0_multi_val"
export PYTHONPATH="$RUN_DIR:$RLLM_DIR/cookbooks/finqa${PYTHONPATH:+:$PYTHONPATH}"
export FINQA_MULTI_TABLE_JUDGE_MODEL=gpt-5.4-nano
# .24's two GPUs cannot complete NCCL P2P collectives reliably.  Both .16 and
# .24 report hostname "tai", so this must be opted into by the .24 launcher
# rather than inferred from hostname.
if [[ "${FINQA_DISABLE_NCCL_P2P:-0}" == "1" ]]; then
  export NCCL_P2P_DISABLE=1
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints"

MODEL_PATH="${FINQA_LOCAL_MODEL_PATH:-/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
TRAIN_MAX_TOKENS_PER_GPU="${FINQA_MULTI_TRAIN_MAX_TOKENS_PER_GPU:-49152}"
[[ -f "$MODEL_PATH/config.json" && -f "$MODEL_PATH/model.safetensors.index.json" ]] || {
  echo "complete local model snapshot not found: $MODEL_PATH" >&2
  exit 1
}
ALGORITHM="${FINQA_MULTI_ALGORITHM:-grpo}"
case "$ALGORITHM" in
  grpo)
    TRAIN_BATCH_SIZE="${FINQA_MULTI_TRAIN_BATCH_SIZE:-8}"
    ROLLOUT_N=8
    PROJECT_NAME=finqa-multitable-v2-grpo
    ALGORITHM_OVERRIDES=(
      algorithm.adv_estimator=grpo
      rllm.algorithm.adv_estimator=grpo
      algorithm.norm_adv_by_std_in_grpo=true
    )
    ;;
  prpo)
    # Preserve the validated single-table PRPO protocol: one rollout per task
    # and a diverse 64-task population baseline.
    TRAIN_BATCH_SIZE="${FINQA_MULTI_TRAIN_BATCH_SIZE:-64}"
    ROLLOUT_N=1
    PROJECT_NAME=finqa-multitable-v2-prpo
    ALGORITHM_OVERRIDES=(
      algorithm.adv_estimator=prpo
      rllm.algorithm.adv_estimator=prpo
      rllm.algorithm.rollout_correction.tis_mode=null
      rllm.async_training.enable=false
      rllm.rejection_sample.min_trajs_per_group=1
    )
    ;;
  *) echo "FINQA_MULTI_ALGORITHM must be grpo or prpo" >&2; exit 2 ;;
esac
RUN_HOST_TAG="${FINQA_MULTI_RUN_HOST_TAG:-$(hostname -s)}"
export FINQA_JUDGE_FINISH_LOG="$RUN_DIR/logs/judge_finish_train_${ALGORITHM}_${RUN_HOST_TAG}.tsv"
[[ "$TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "invalid FINQA_MULTI_TRAIN_BATCH_SIZE" >&2; exit 2; }
STEPS_PER_EPOCH=$((991 / TRAIN_BATCH_SIZE))
(( STEPS_PER_EPOCH > 0 )) || { echo "train batch exceeds multi_train size" >&2; exit 2; }

case "$MODE" in
  smoke)
    TOTAL_EPOCHS=1; TOTAL_STEPS=2; SAVE_FREQ=-1
    EXPERIMENT_NAME="finqa-multitable-v2-${ALGORITHM}-smoke-${RUN_HOST_TAG}"; RESUME_MODE=disable
    ;;
  epoch)
    TOTAL_EPOCHS="${FINQA_MULTI_EPOCHS:-50}"
    [[ "$TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid FINQA_MULTI_EPOCHS" >&2; exit 2; }
    TOTAL_STEPS=$((STEPS_PER_EPOCH * TOTAL_EPOCHS)); SAVE_FREQ=$STEPS_PER_EPOCH
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-finqa-multitable-v2-${ALGORITHM}-b${TRAIN_BATCH_SIZE}}"
    RESUME_MODE=auto
    ;;
  *) echo "usage: $0 {smoke|epoch} [hydra overrides...]" >&2; exit 2 ;;
esac

cd "$RLLM_DIR/cookbooks/finqa"
exec python -u "$RUN_DIR/train_multitable.py" \
    rllm/backend=verl \
    "${ALGORITHM_OVERRIDES[@]}" \
    rllm.algorithm.use_rllm=true \
    +model.name="$MODEL_PATH" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.lora.merge=True \
    `# Multi-table trajectories exceed the memory envelope of materialised` \
    `# [tokens,vocab] logits when entropy is enabled.  This is veRL's public` \
    `# fused log-prob/entropy path and is applied identically to both arms.` \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
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
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    rllm.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${FINQA_MULTI_GPU_MEMORY_UTILIZATION:-0.35}" \
    actor_rollout_ref.rollout.max_model_len=49152 \
    `# This is the dynamic micro-batch packing ceiling.  It must be at least` \
    `# the longest whole trajectory because veRL never splits one sequence.` \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$TRAIN_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$TRAIN_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$TRAIN_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    rllm.data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size=8 \
    data.max_prompt_length=40960 \
    data.max_response_length=8192 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.logger="['console']" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    rllm.trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.total_training_steps="$TOTAL_STEPS" \
    rllm.trainer.total_batches="$TOTAL_STEPS" \
    trainer.test_freq=-1 \
    rllm.trainer.test_freq=-1 \
    trainer.save_freq="$SAVE_FREQ" \
    rllm.trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train=false \
    rllm.trainer.val_before_train=false \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$RUN_DIR/checkpoints/\${trainer.experiment_name}" \
    "$@"

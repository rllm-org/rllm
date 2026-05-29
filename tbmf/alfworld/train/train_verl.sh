#!/usr/bin/env bash
# Train the FrozenLake agent with the verl (distributed GPU) backend.
#
# Prerequisites:
#   1. Install rllm with verl extras:     uv pip install -e ".[verl]"
#   2. Install this cookbook:              uv pip install --no-deps -e cookbooks/frozenlake
#   3. Generate the dataset:               uv run python cookbooks/frozenlake/prepare_frozenlake_data.py

set -euo pipefail

unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
# case ",${RLLM_EXCLUDE:-}," in
#     *,CUDA_VISIBLE_DEVICES,*) ;;
#     *) export RLLM_EXCLUDE="${RLLM_EXCLUDE:+$RLLM_EXCLUDE,}CUDA_VISIBLE_DEVICES" ;;
# esac
# vLLM 0.17 does not read VLLM_ATTENTION_BACKEND; pass attention_backend
# through actor_rollout_ref.rollout.engine_kwargs.vllm below instead.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
# The site environment has an optional deep_gemm wheel that is ABI-incompatible
# with the active Torch build. Disable that optional vLLM path for this BF16 run.
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN
# Keep the full CUDA mask in Ray workers and let verl choose the device from
# Ray's accelerator id. This avoids both FSDP ranks initializing on cuda:0.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# cd "$(dirname "$0")/../../rllm"
export LD_LIBRARY_PATH="/home/sankuai/conda/lib:$LD_LIBRARY_PATH"
MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/wangyu363/model/Qwen3-4B-Instruct-2507

echo "which python: $(which python3)"

CUDA_VISIBLE_DEVICES=0,1 python3 -m tbmf.alfworld.train.train_grpo \
    rllm/backend=verl \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    rllm.algorithm.use_rllm=true \
    data.train_batch_size=16 \
    data.val_batch_size=50 \
    data.max_prompt_length=12000 \
    data.max_response_length=4096 \
    +model.name=$MODEL_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.mode=0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=NONE \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    +actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.logger="['console','ui']" \
    trainer.project_name=alfworld \
    trainer.experiment_name=qwen3-4b-instruct-verl \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=disable \
    "$@"

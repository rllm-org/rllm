#!/usr/bin/env bash
# Train MathToolWorkflow on the verl backend (4xH100 smoke run).
#
# Goal: 1 validation step (val_before_train) + 2 training steps end-to-end.
# Success criteria:
#   * No crashes.
#   * `batch/merge_compression_ratio > 1.0` shows up in the log on step 1.
#
# Prereqs:
#   uv pip install -e ".[verl]"
#   uv pip install --no-deps -e cookbooks/math_tool_agent
#   rllm dataset pull deepscaler_math && rllm dataset pull math500

set -euo pipefail

unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
export VLLM_USE_V1=${VLLM_USE_V1:-1}

MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507

cd "$(dirname "$0")"

python -u train_workflow.py \
    rllm/backend=verl \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    data.train_batch_size=8 \
    data.val_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    +model.name=$MODEL_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.lora.rank=32 \
    +actor_rollout_ref.model.lora.alpha=32 \
    +actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
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
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
    actor_rollout_ref.rollout.enforce_eager=False \
    +actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    +rllm.workflow.max_turns=4 \
    rllm.workflow.n_parallel_tasks=32 \
    trainer.logger="['console']" \
    trainer.project_name=math_tool_workflow_smoke \
    trainer.experiment_name=qwen3-4b-instruct-verl-smoke \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=999 \
    trainer.test_freq=999 \
    trainer.total_epochs=1 \
    rllm.trainer.total_batches=2 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=disable \
    "$@"

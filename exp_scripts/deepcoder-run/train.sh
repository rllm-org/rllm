#!/usr/bin/env bash
# Experiment wrapper: upstream DeepCoder flow/evaluator/trainer are unchanged.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
RLLM=/home/yanan/agents/rllm
source "$ROOT/../finqa-grpo-run/env.sh"
source "$VENV/bin/activate"
export RLLM_HOME="$ROOT/runtime"
export PYTHONPATH="$ROOT:$RLLM/cookbooks/deepcoder:$RLLM${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARM="${1:?usage: train.sh grpo|prpo [hydra overrides]}"
shift
case "$ARM" in
  grpo) BATCH="${DEEPCODER_GRPO_TASK_BATCH:-16}"; N=8; HOST=16 ;;
  prpo) BATCH="${DEEPCODER_PRPO_TASK_BATCH:-64}"; N=1; HOST="${DEEPCODER_HOST:-24}"; [[ "$HOST" == 24 ]] && export NCCL_P2P_DISABLE=1 ;;
  *) exit 2 ;;
esac
NAME="${DEEPCODER_RUN_NAME:-deepcoder-${ARM}-b${BATCH}-n${N}-${HOST}-forkserver}"
EPOCHS="${DEEPCODER_EPOCHS:-30}"
if [[ " $* " == *" --cfg "* ]]; then
  ROWS=24287 # Hydra configuration inspection only; upstream main is not run.
else
  ROWS=$(python -c 'import os,pyarrow.parquet as p; n=p.ParquetFile(os.path.join(os.environ["RLLM_HOME"],"datasets/deepcoder/train.parquet")).metadata.num_rows; assert n==24287; print(n)')
fi
STEPS=$((ROWS / BATCH))
if [[ "$ARM" == grpo ]]; then
  # GRPO uses 16 distinct tasks per optimizer step.  Saving every 64 steps
  # therefore snapshots the policy after exactly 1,024 training tasks.
  # Keep DEEPCODER_SAVE_FREQ as an explicit escape hatch for smoke tests.
  SAVE_FREQ="${DEEPCODER_SAVE_FREQ:-64}"
else
  SAVE_FREQ="${DEEPCODER_SAVE_FREQ:-$STEPS}"
fi
PARALLEL_TASKS="${DEEPCODER_PARALLEL_TASKS:-256}"
MODEL=/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
test -f "$MODEL/model.safetensors.index.json"
# Checkpoints are machine-local by default.  The repository lives on NFS, but
# checkpoint writes must not put fsync/network failures on the training path.
# Evaluation/archival copies are staged asynchronously after a checkpoint is
# complete.  An explicit override remains available for unusual deployments.
CHECKPOINT_ROOT="${DEEPCODER_CHECKPOINT_ROOT:-/home/yanan/.deepcoder-checkpoints}"
mkdir -p "$CHECKPOINT_ROOT/$NAME" "$ROOT/logs"
echo "ARM=$ARM tasks=$ROWS batch=$BATCH n=$N rollouts_per_step=$((BATCH*N)) epoch_steps=$STEPS epochs=$EPOCHS save_freq=$SAVE_FREQ parallel_tasks=$PARALLEL_TASKS shuffle=true"
cd "$RLLM"
exec python -u "$ROOT/train_compatible.py" \
  rllm/backend=verl algorithm.adv_estimator="$ARM" rllm.algorithm.adv_estimator="$ARM" \
  algorithm.norm_adv_by_std_in_grpo=true rllm.algorithm.use_rllm=true \
  rllm.async_training.enable=false rllm.algorithm.rollout_correction.tis_mode=null \
  rllm.workflow.n_parallel_tasks="$PARALLEL_TASKS" \
  data.train_batch_size="$BATCH" rllm.data.train_batch_size="$BATCH" \
  data.val_batch_size=64 data.shuffle=true data.seed=1234 rllm.data.seed=1234 \
  data.max_prompt_length=8192 data.max_response_length=16384 \
  +model.name="$MODEL" actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.lora_rank=32 actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.lora.merge=true actor_rollout_ref.hybrid_engine=true \
  actor_rollout_ref.model.use_fused_kernels=true actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.enforce_eager=false actor_rollout_ref.rollout.max_model_len=32768 \
  actor_rollout_ref.rollout.temperature=0.6 rllm.rollout.train.temperature=0.6 rllm.rollout.train.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${DEEPCODER_GPU_MEMORY_UTILIZATION:-0.9}" \
  actor_rollout_ref.rollout.n="$N" rllm.rollout.n="$N" rllm.rejection_sample.min_trajs_per_group=1 \
  actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.logger="['console']" trainer.project_name=deepcoder-comparison trainer.experiment_name="$NAME" \
  trainer.n_gpus_per_node=2 trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" rllm.trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq=-1 rllm.trainer.test_freq=-1 trainer.val_before_train=false rllm.trainer.val_before_train=false \
  trainer.total_epochs="$EPOCHS" rllm.trainer.total_epochs="$EPOCHS" \
  trainer.total_training_steps="$((STEPS*EPOCHS))" rllm.trainer.total_batches="$((STEPS*EPOCHS))" \
  trainer.default_hdfs_dir=null trainer.default_local_dir="$CHECKPOINT_ROOT/$NAME" trainer.resume_mode=disable \
  "$@"

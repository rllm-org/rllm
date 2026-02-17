#!/bin/bash
# SWE RL Training Script
# Usage: bash swe-train-rl.sh <model> [root]
#
# Examples:
#   # Train from base model
#   bash swe-train-rl.sh Qwen3-8B
#
#   # Train from SFT checkpoint (pass full path as model)
#   bash swe-train-rl.sh experiments/verl/agentic-swe-sft/global_step_1024
#
#   # Train with custom root directory
#   bash swe-train-rl.sh Qwen3-8B /mnt/bn/my-bucket

set -x

# ============ Arguments ============
MODEL_NAME=${1:?'Usage: bash swe-train-rl.sh <model_name> [root_dir]'}
ROOT_DIR=${2:-'/mnt/bn/trae-research-models/xujunjielong'}

# ============ Environment ============
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:false"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export GLOO_SOCKET_IFNAME='eth0'
export NCCL_SOCKET_IFNAME='eth0'

export UV_INDEX_URL=https://bytedpypi.byted.org/simple/
export HF_ENDPOINT=https://hf-mirror.com
export DOCKER_MIRROR_PREFIX='aibrix-docker-mirror-cn-beijing.cr.volces.com'

export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

export ARL_GATEWAY_URL="http://14.103.184.145:8080"

# ============ Config ============
WAND_PROJECT='xujunjielong'
EXPERIMENT_NAME='agentic-swe-rl'

# ============ Byted env ============
uv pip uninstall ray wandb bytedray byted-wandb
uv pip install bytedray[default,data,serve,bytedance] byted-wandb

# If use fsdp offload, please set:
# tensor_model_parallel=$ARNOLD_WORKER_NUM
# actor_rollout_ref.actor.fsdp_config.param_offload=true
# actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
# actor_rollout_ref.ref.fsdp_config.param_offload=true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/scripts/patch_verl.sh"

# ============ Run Training ============
uv run --no-sync python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=data/swe/R2E_Gym_Subset.parquet \
    data.val_files=data/swe/SWE_Bench_Verified.parquet \
    trainer.default_local_dir=$ROOT_DIR/experiments/verl/$EXPERIMENT_NAME \
    trainer.rollout_data_dir=$ROOT_DIR/rollouts/$EXPERIMENT_NAME \
    data.train_batch_size=8 \
    data.val_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=32 \
    actor_rollout_ref.model.path=$ROOT_DIR/models/$MODEL_NAME \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=false \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${ARNOLD_WORKER_NUM:-1} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    rllm.env.name=swe \
    rllm.agent.name=sweagent \
    rllm.agent.max_steps=50 \
    rllm.agent.overlong_filter=true \
    rllm.agent.trajectory_timeout=1200 \
    +rllm.env.env_args.verbose=false \
    +rllm.env.env_args.scaffold=r2egym \
    +rllm.agent.agent_args.scaffold=r2egym \
    trainer.total_epochs=1000 \
    2>&1 | tee $EXPERIMENT_NAME.log

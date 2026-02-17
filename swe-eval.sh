#!/bin/bash
# SWE Evaluation Script
# Usage: bash swe-eval.sh <model_path> [val_files] [n_samples]
#
# Examples:
#   # Evaluate base model (greedy, n=1)
#   bash swe-eval.sh /mnt/bn/trae-research-models/xujunjielong/models/Qwen3-8B
#
#   # Evaluate SFT checkpoint
#   bash swe-eval.sh /mnt/bn/trae-research-models/xujunjielong/experiments/verl/agentic-swe-sft/global_step_1024
#
#   # Evaluate RL checkpoint with pass@5
#   bash swe-eval.sh /mnt/bn/trae-research-models/xujunjielong/experiments/verl/agentic-swe-rl/global_step_100 data/swe/SWE_Bench_Verified.parquet 5

set -x

# ============ Arguments ============
MODEL_NAME=${1:?'Usage: bash swe-eval.sh <model_name> [n_samples] [root_dir]'}
N_SAMPLES=${2:-1}
ROOT_DIR=${3:-'/mnt/bn/trae-research-models/xujunjielong'}

# Temperature: greedy (0) for n=1, sampling (1.0) for n>1
if [ "$N_SAMPLES" -gt 1 ]; then
    TEMPERATURE=1.0
    DO_SAMPLE=true
else
    TEMPERATURE=0
    DO_SAMPLE=false
fi

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
EXPERIMENT_NAME='agentic-swe-eval'

# ============ Byted env ============
uv pip uninstall ray wandb bytedray byted-wandb
uv pip install bytedray[default,data,serve,bytedance] byted-wandb

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/scripts/patch_verl.sh"

# ============ Run Training ============
uv run --no-sync python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=data/swe/SWE_Bench_Verified.parquet \
    data.val_files=data/swe/SWE_Bench_Verified.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=32 \
    actor_rollout_ref.model.path=$ROOT_DIR/models/$MODEL_NAME \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=$N_SAMPLES \
    actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$DO_SAMPLE \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.val_before_train=true \
    trainer.val_only=true \
    trainer.critic_warmup=0 \
    trainer.total_epochs=1 \
    trainer.save_freq=999999 \
    trainer.test_freq=999999 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${ARNOLD_WORKER_NUM:-1} \
    trainer.default_local_dir=$ROOT_DIR/experiments/verl/$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    rllm.env.name=swe \
    rllm.agent.name=sweagent \
    rllm.agent.max_steps=100 \
    rllm.agent.overlong_filter=false \
    rllm.agent.trajectory_timeout=1200 \
    +rllm.env.env_args.verbose=false \
    +rllm.env.env_args.scaffold=r2egym \
    +rllm.agent.agent_args.scaffold=r2egym \
    2>&1 | tee $EXPERIMENT_NAME.log

# ============ Generate Report ============
RESULTS_FILE="$ROOT_DIR/experiments/verl/$EXPERIMENT_NAME/val_results/step_0.jsonl"
if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "Generating evaluation report..."
    uv run --no-sync python3 "$SCRIPT_DIR/scripts/swe_report.py" "$RESULTS_FILE"
else
    echo "Warning: results file not found at $RESULTS_FILE"
fi

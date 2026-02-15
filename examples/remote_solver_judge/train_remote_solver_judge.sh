#!/usr/bin/env bash
# =============================================================================
# Remote Solver-Judge Training Script
#
# This demonstrates online RL training with remote agent episode generation.
#
# Architecture:
#   [Trainer]                                 [Remote Agent Server(s)]
#       |                                           |
#       |--- exposes inference API (port 8089) ---->|
#       |<-- POST /generate_episode (episodes) -----|
#       |                                           |
#   (update policy)                       (runs SolverJudgeWorkflow
#                                          using trainer's inference API)
#
# Steps:
#   1. Start one or more remote agent servers (in separate terminals/containers).
#   2. Run this script to launch training.
#
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-4B-Instruct-2507"}
MODEL_LORA_RANK=${MODEL_LORA_RANK:-32}
TOTAL_BATCHES=${TOTAL_BATCHES:-100}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

# Remote agent endpoints (comma-separated if multiple)
REMOTE_AGENT_ENDPOINTS=${REMOTE_AGENT_ENDPOINTS:-'["http://localhost:5100"]'}
INFERENCE_API_PORT=${INFERENCE_API_PORT:-8089}

# Run name
date_str=$(date +%Y-%m-%d)
time_str=$(date +%H-%M-%S)
run_name="REMOTE_SOLVER_JUDGE_${date_str}_${time_str}"
local_dir="${LOCAL_DIR:-/tmp/rllm_runs/${run_name}}"

# ---------------------------------------------------------------------------
# Step 1: Remind user to start the remote agent server
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Remote Solver-Judge Training"
echo "============================================================"
echo ""
echo "Make sure the remote agent server is running:"
echo "  python -m examples.remote_solver_judge.remote_agent_server --port 5100"
echo ""
echo "Trainer inference API will be exposed on port ${INFERENCE_API_PORT}"
echo "Remote agent endpoints: ${REMOTE_AGENT_ENDPOINTS}"
echo ""
echo "Starting training in 3 seconds..."
sleep 3

# ---------------------------------------------------------------------------
# Step 2: Launch training with remote agent mode enabled
# ---------------------------------------------------------------------------
python3 -m examples.remote_solver_judge.train_remote_solver_judge \
    rllm/backend=tinker \
    \
    rllm.remote_agent.enabled=true \
    "rllm.remote_agent.endpoints=${REMOTE_AGENT_ENDPOINTS}" \
    rllm.remote_agent.inference_api.port=${INFERENCE_API_PORT} \
    rllm.remote_agent.timeout=600 \
    rllm.remote_agent.max_concurrent=128 \
    \
    rllm.compact_filtering.enable=False \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.use_rllm=true \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.trainer.total_batches=$TOTAL_BATCHES \
    rllm.trainer.total_epochs=$TOTAL_EPOCHS \
    rllm.trainer.logger='[console]' \
    rllm.trainer.project_name='remote-solver-judge' \
    rllm.trainer.experiment_name=$run_name \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=20 \
    rllm.trainer.save_freq=20 \
    rllm.rollout.n=4 \
    rllm.rollout.n_val=1 \
    \
    model.name=$MODEL_PATH \
    model.lora_rank=$MODEL_LORA_RANK \
    training.group_size=5 \
    training.learning_rate=4e-5 \
    training.default_local_dir=$local_dir \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.train_batch_size=32 \
    data.val_batch_size=512

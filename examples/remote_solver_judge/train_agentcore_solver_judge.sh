#!/usr/bin/env bash
# =============================================================================
# AgentCore Remote Training Script
#
# Online RL training using Amazon Bedrock AgentCore Runtime (ACR) as the
# remote agent runtime with fire-and-forget episode collection via S3/SQS.
#
# Architecture:
#   [rLLM Trainer]                          [ACR Agent Sessions]
#       |                                         |
#       |--- exposes inference API (port 8089) -->|
#       |                                         |
#       |--- invoke_agent_runtime (fire) -------->|
#       |<-- {"status": "processing"}             |
#       |                                         |--- agent runs in background
#       |                                         |--- calls /v1/model_response
#       |                                         |--- saves rollout to S3
#       |                                         |--- notifies SQS
#       |                                         |
#       |<-- poll SQS for completions             |
#       |<-- download episodes from S3            |
#       |                                         |
#   (update policy)
#
# Prerequisites:
#   1. Deploy agent to ACR:
#        cd agentcore-rl-toolkit/examples/strands_math_agent
#        agentcore deploy --agent strands_math_agent_rl_tokens \
#            --env BASE_URL=http://<TRAINER_IP>:8089/v1 \
#            --env MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
#
#   2. Create S3 bucket and SQS queue:
#        aws s3 mb s3://agentcore-rl
#        aws sqs create-queue --queue-name agentcore-rl
#
#   3. Grant IAM permissions (see agentcore-rl-toolkit README for details).
#
# Usage:
#   # Minimal (two required env vars):
#   AGENT_RUNTIME_ARN="arn:aws:bedrock-agentcore:us-west-2:123456789:runtime/my-agent-abc123" \
#   SQS_URL="https://sqs.us-west-2.amazonaws.com/123456789/agentcore-rl" \
#   bash examples/remote_solver_judge/train_agentcore_solver_judge.sh
#
#   # Override key settings:
#   AGENT_RUNTIME_ARN="arn:aws:bedrock-agentcore:us-west-2:123456789:runtime/my-agent-abc123" \
#   SQS_URL="https://sqs.us-west-2.amazonaws.com/123456789/my-queue" \
#   S3_BUCKET="my-bucket" \
#   MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507" \
#   bash examples/remote_solver_judge/train_agentcore_solver_judge.sh
#
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-4B-Instruct-2507"}
MODEL_LORA_RANK=${MODEL_LORA_RANK:-32}

# Training
TOTAL_BATCHES=${TOTAL_BATCHES:-100}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-512}
ROLLOUT_N=${ROLLOUT_N:-4}
LEARNING_RATE=${LEARNING_RATE:-4e-5}

# Remote agent (AgentCore)
AGENT_RUNTIME_ARN=${AGENT_RUNTIME_ARN:-""}
INFERENCE_API_PORT=${INFERENCE_API_PORT:-8089}
MAX_CONCURRENT=${MAX_CONCURRENT:-128}
RETRY_LIMIT=${RETRY_LIMIT:-3}

# AgentCore S3/SQS
S3_BUCKET=${S3_BUCKET:-"agentcore-rl"}
SQS_URL=${SQS_URL:-""}
ACR_TIMEOUT=${ACR_TIMEOUT:-1800}
POLL_INTERVAL=${POLL_INTERVAL:-2.0}
SQS_BATCH_SIZE=${SQS_BATCH_SIZE:-10}

# Run naming
date_str=$(date +%Y-%m-%d)
time_str=$(date +%H-%M-%S)
run_name="AGENTCORE_SOLVER_JUDGE_${date_str}_${time_str}"
local_dir="${LOCAL_DIR:-/tmp/rllm_runs/${run_name}}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "$AGENT_RUNTIME_ARN" ]]; then
    echo "ERROR: AGENT_RUNTIME_ARN is required."
    echo ""
    echo "  export AGENT_RUNTIME_ARN=\"arn:aws:bedrock-agentcore:<region>:<account>:runtime/<name>\""
    echo ""
    echo "  You can find it with:  agentcore list"
    exit 1
fi

if [[ -z "$SQS_URL" ]]; then
    echo "ERROR: SQS_URL is required."
    echo ""
    echo "  export SQS_URL=\"https://sqs.<region>.amazonaws.com/<account>/agentcore-rl\""
    echo ""
    echo "  You can find it with:  aws sqs get-queue-url --queue-name agentcore-rl"
    exit 1
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo " AgentCore Remote Training"
echo "============================================================"
echo ""
echo "  Model:           ${MODEL_PATH}"
echo "  LoRA rank:       ${MODEL_LORA_RANK}"
echo "  Batches:         ${TOTAL_BATCHES}"
echo "  Rollouts/prompt: ${ROLLOUT_N}"
echo ""
echo "  Agent ARN:       ${AGENT_RUNTIME_ARN}"
echo "  Inference API:   0.0.0.0:${INFERENCE_API_PORT}"
echo "  S3 bucket:       ${S3_BUCKET}"
echo "  SQS URL:         ${SQS_URL}"
echo "  ACR timeout:     ${ACR_TIMEOUT}s"
echo "  Max concurrent:  ${MAX_CONCURRENT}"
echo ""
echo "  Run name:        ${run_name}"
echo "  Output dir:      ${local_dir}"
echo ""
echo "Starting training in 3 seconds..."
sleep 3

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
python3 -m examples.remote_solver_judge.train_remote_solver_judge \
    rllm/backend=tinker \
    \
    rllm.remote_agent.enabled=true \
    rllm.remote_agent.mode=agentcore \
    rllm.remote_agent.inference_api.port=${INFERENCE_API_PORT} \
    rllm.remote_agent.max_concurrent=${MAX_CONCURRENT} \
    rllm.remote_agent.retry_limit=${RETRY_LIMIT} \
    rllm.remote_agent.agentcore.agent_runtime_arn=${AGENT_RUNTIME_ARN} \
    rllm.remote_agent.agentcore.s3_bucket=${S3_BUCKET} \
    rllm.remote_agent.agentcore.sqs_url=${SQS_URL} \
    rllm.remote_agent.agentcore.exp_id=${run_name} \
    rllm.remote_agent.agentcore.timeout=${ACR_TIMEOUT} \
    rllm.remote_agent.agentcore.poll_interval=${POLL_INTERVAL} \
    rllm.remote_agent.agentcore.batch_size=${SQS_BATCH_SIZE} \
    \
    rllm.compact_filtering.enable=False \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.use_rllm=true \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.trainer.total_batches=${TOTAL_BATCHES} \
    rllm.trainer.total_epochs=${TOTAL_EPOCHS} \
    rllm.trainer.logger='[console]' \
    rllm.trainer.project_name='agentcore-solver-judge' \
    rllm.trainer.experiment_name=${run_name} \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=20 \
    rllm.trainer.save_freq=20 \
    rllm.rollout.n=${ROLLOUT_N} \
    rllm.rollout.n_val=1 \
    \
    model.name=${MODEL_PATH} \
    model.lora_rank=${MODEL_LORA_RANK} \
    training.group_size=5 \
    training.learning_rate=${LEARNING_RATE} \
    training.default_local_dir=${local_dir} \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE}

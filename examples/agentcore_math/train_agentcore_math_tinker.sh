#!/usr/bin/env bash
set -x

# Load environment variables (TINKER_API_KEY, AGENTCORE_AGENT_ARN, AGENTCORE_S3_BUCKET)
set -a && source .env && set +a

MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507

python -m examples.agentcore_math.train_agentcore_math_tinker \
    rllm/backend=tinker \
    model.name=$MODEL_PATH \
    model.lora_rank=16 \
    training.group_size=4 \
    validation.group_size=1 \
    training.learning_rate=2e-5 \
    training.max_length=32768 \
    sampling.train.temperature=1.0 \
    sampling.val.temperature=0.6 \
    data.max_prompt_length=30720 \
    data.max_response_length=2048 \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger="['console', 'ui']" \
    rllm.trainer.project_name=agentcore-math \
    rllm.trainer.experiment_name=gsm8k-agentcore-tinker-4b-instruct \
    rllm.trainer.val_before_train=true \
    rllm.trainer.test_freq=10 \
    rllm.trainer.save_freq=100 \
    rllm.remote_runtime.enabled=true \
    rllm.remote_runtime.backend=agentcore \
    rllm.remote_runtime.agentcore.agent_runtime_arn=$AGENTCORE_AGENT_ARN \
    rllm.remote_runtime.agentcore.s3_bucket=$AGENTCORE_S3_BUCKET \
    rllm.remote_runtime.agentcore.tps_limit=25 \
    rllm.remote_runtime.session_timeout=300 \

#!/bin/bash
set -x

# Minimal test script for verifiers training
# Reduced parallelism and batch sizes for quick debugging

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
    export TINKER_API_KEY
fi

python3 -m examples.verifiers_env.train \
    --config-name=tinker_rl_trainer \
    +backend=tinker \
    tinker_base_url=null \
    model.name="Qwen/Qwen3-4B-Instruct-2507" \
    model.lora_rank=16 \
    +verifiers.env_id="primeintellect/alphabet-sort" \
    +verifiers.max_samples=20 \
    algorithm.adv_estimator=grpo \
    training.group_size=4 \
    training.learning_rate=2e-5 \
    training.max_length=4096 \
    sampling.temperature=1.0 \
    sampling.top_p=1.0 \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    workflow.n_parallel_tasks=4 \
    workflow.retry_limit=1 \
    trainer.total_epochs=1 \
    'trainer.logger=[console,wandb]' \
    trainer.project_name="verifiers-test" \
    trainer.experiment_name="test-run" \
    trainer.test_freq=1 \
    trainer.save_freq=100 \
    trainer.val_before_train=false

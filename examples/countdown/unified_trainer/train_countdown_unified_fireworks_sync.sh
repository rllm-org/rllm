set -x

# Requires FIREWORKS_API_KEY to be set. The trainer job and inference
# deployment are provisioned on Fireworks at startup (via the cookbook's
# training.provision API) and torn down on shutdown.

python -m examples.countdown.unified_trainer.train_countdown_unified_fireworks \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/qwen3p5-35b-a3b \
    model.tokenizer_model=Qwen/Qwen3.5-35B-A3B \
    model.lora_rank=32 \
    fireworks_config.policy_trainer_shape_id=accounts/fireworks/trainingShapes/qwen3p5-35b-a3b-256k-lora \
    rollout_engine.reasoning_effort=none \
    training.group_size=8 \
    training.learning_rate=1e-5 \
    training.max_length=8192 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.rollout.val.temperature=1.0 \
    rllm.rollout.val.top_p=1.0 \
    validation.group_size=1 \
    rllm.workflow.n_parallel_tasks=256 \
    rllm.workflow.retry_limit=1 \
    rllm.workflow.raise_on_error=false \
    data.max_prompt_length=2048 \
    data.max_response_length=6144 \
    rllm.data.max_prompt_length=2048 \
    rllm.data.max_response_length=6144 \
    data.train_batch_size=32 \
    data.val_batch_size=1024 \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.async_training.enable=false \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='rllm-countdown' \
    rllm.trainer.experiment_name='countdown-fireworks-Qwen3.5-35B-A3B-sync' \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=0 \
    rllm.trainer.save_freq=5

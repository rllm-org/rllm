set -x

# Requires FIREWORKS_API_KEY to be set. The trainer job and inference
# deployment are provisioned on Fireworks at startup (via the cookbook's
# training.provision API) and torn down on shutdown.

python -m examples.countdown.unified_trainer.train_countdown_unified_fireworks \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/qwen3-4b \
    model.lora_rank=0 \
    training_infra.training_shape_id=qwen3-4b-minimum-h200 \
    deployment.tokenizer_model=Qwen/Qwen3-4B \
    training.group_size=8 \
    training.learning_rate=1e-5 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.rollout.val.temperature=1.0 \
    rllm.rollout.val.top_p=1.0 \
    validation.group_size=1 \
    rllm.workflow.n_parallel_tasks=256 \
    rllm.workflow.retry_limit=1 \
    rllm.workflow.raise_on_error=false \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.train_batch_size=1 \
    data.val_batch_size=1024 \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.async_training.enable=true \
    rllm.async_training.mini_batch_size=32 \
    rllm.async_training.fwd_bwd_group_size=8 \
    rllm.async_training.staleness_threshold=0.5 \
    rllm.async_training.trigger_parameter_sync_step=1 \
    rllm.async_training.partial_rollout=true \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='rllm-countdown' \
    rllm.trainer.experiment_name='countdown-fireworks-async-staleness-0.5' \
    rllm.trainer.val_before_train=true \
    rllm.trainer.test_freq=10 \
    rllm.trainer.save_freq=-1

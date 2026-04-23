set -x

python -m examples.countdown.unified_trainer.train_countdown_unified_tinker \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-8B \
    model.lora_rank=32 \
    training.group_size=8 \
    training.learning_rate=2e-5 \
    training.max_length=4096 \
    sampling.train.temperature=1.0 \
    sampling.train.top_p=1.0 \
    sampling.val.temperature=0.6 \
    sampling.val.top_p=0.95 \
    validation.group_size=1 \
    rllm.workflow.n_parallel_tasks=256 \
    rllm.workflow.retry_limit=1 \
    rllm.workflow.raise_on_error=false \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.train_batch_size=64 \
    data.val_batch_size=1024 \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.async_training.enable=false \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='rllm-countdown' \
    rllm.trainer.experiment_name='countdown-tinker-sync' \
    rllm.trainer.val_before_train=true \
    rllm.trainer.test_freq=10 \
    rllm.trainer.save_freq=-1

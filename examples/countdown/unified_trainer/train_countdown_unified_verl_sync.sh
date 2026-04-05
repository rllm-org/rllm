set -x

python -m examples.countdown.unified_trainer.train_countdown_unified_verl \
    rllm/backend=verl \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.rollout.name=vllm \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    rllm.rollout.n=8 \
    rllm.rollout.n_val=1 \
    rllm.rollout.sampling.train.temperature=1.0 \
    rllm.rollout.sampling.train.top_p=1.0 \
    rllm.rollout.sampling.val.temperature=1.0 \
    rllm.rollout.sampling.val.top_p=1.0 \
    rllm.workflow.n_parallel_tasks=256 \
    rllm.workflow.retry_limit=1 \
    rllm.workflow.raise_on_error=false \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.train_batch_size=32 \
    data.val_batch_size=1024 \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.async_training.enable=false \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger='[wandb]' \
    rllm.trainer.project_name='rllm-countdown' \
    rllm.trainer.experiment_name='countdown-verl-sync' \
    rllm.trainer.val_before_train=true \
    rllm.trainer.test_freq=10 \
    rllm.trainer.save_freq=-1

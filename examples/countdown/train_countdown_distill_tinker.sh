set -x

python -m examples.countdown.train_countdown_tinker \
    model.name=Qwen/Qwen3-8B \
    model.lora_rank=32 \
    training.group_size=8 \
    training.learning_rate=4e-5 \
    sampling.temperature=1.0 \
    sampling.top_p=1.0 \
    algorithm.adv_estimator=distill \
    algorithm.shared_tokenizer=True \
    algorithm.teacher_rollout_args.model=kylemontgomery/countdown-solver-8b \
    algorithm.teacher_rollout_args.base_url=http://localhost:32000/v1 \
    algorithm.teacher_rollout_args.api_key=EMPTY \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.train_batch_size=64 \
    data.val_batch_size=1024 \
    trainer.total_epochs=100 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='countdown-distill-tinker-8b' \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=1000 \
    trainer.default_local_dir='./outputs/countdown-distill-tinker-8b' \
    rollout_engine.bypass_render_with_parser=True

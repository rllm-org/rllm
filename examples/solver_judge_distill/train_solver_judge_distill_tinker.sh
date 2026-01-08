set -x

python -m examples.solver_judge_distill.train_solver_judge_distill_tinker \
    model.name=Qwen/Qwen3-8B \
    model.lora_rank=32 \
    training.group_size=1 \
    training.val_group_size=4 \
    training.learning_rate=5e-4 \
    sampling.temperature=1.0 \
    sampling.top_p=1.0 \
    algorithm.adv_estimator=distill \
    algorithm.shared_tokenizer=False \
    algorithm.teacher_rollout_args.backend=tinker \
    algorithm.teacher_rollout_args.model=moonshotai/Kimi-K2-Thinking\
    data.max_prompt_length=32768 \
    data.max_response_length=32768 \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    trainer.total_epochs=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='solver-judge-distill' \
    trainer.experiment_name='solver-judge-math-distill-8b-tinker-kimi-tinker' \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.default_local_dir='./outputs/solver-judge-math-distill-8b-tinker-kimi-tinker' \
    rollout_engine.bypass_render_with_parser=True \
    workflow.n_parallel_tasks=256 \
    training.rollout_timeout=1200

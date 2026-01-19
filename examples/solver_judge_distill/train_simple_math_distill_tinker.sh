set -x

python -m examples.solver_judge_distill.train_simple_math_distill_tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    training.group_size=1 \
    training.val_group_size=4 \
    training.learning_rate=4e-5 \
    sampling.temperature=1.0 \
    sampling.top_p=1.0 \
    algorithm.adv_estimator=distill \
    algorithm.shared_tokenizer=False \
    algorithm.teacher_rollout_args.backend=tinker \
    algorithm.teacher_rollout_args.model=Qwen/Qwen3-235B-A22B-Instruct-2507 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    trainer.total_epochs=100 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='solver-judge-distill' \
    trainer.experiment_name='simple-math-distill-3b-235b-instruct' \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=20 \
    trainer.default_local_dir='./outputs/simple-math-distill-3b-235b-instruct' \
    rollout_engine.bypass_render_with_parser=True \
    workflow.n_parallel_tasks=256 \
    training.rollout_timeout=1200


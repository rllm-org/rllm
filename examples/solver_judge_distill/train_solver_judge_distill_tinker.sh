set -x

python -m examples.solver_judge_distill.train_solver_judge_distill_tinker \
    model.name=Qwen/Qwen3-8B \
    model.lora_rank=32 \
    training.group_size=8 \
    training.learning_rate=4e-5 \
    sampling.temperature=1.0 \
    sampling.top_p=1.0 \
    algorithm.adv_estimator=distill \
    algorithm.shared_tokenizer=True \
    algorithm.teacher_rollout_args.model=Qwen/Qwen3-32B \
    algorithm.teacher_rollout_args.base_url=http://localhost:32000/v1 \
    algorithm.teacher_rollout_args.api_key=EMPTY \
    data.max_prompt_length=4096 \
    data.max_response_length=32768 \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    trainer.total_epochs=100 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='solver-judge-distill' \
    trainer.experiment_name='solver-judge-math-distill-8b' \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=5 \
    trainer.default_local_dir='./outputs/solver-judge-distill' \
    rollout_engine.bypass_render_with_parser=True

set -x

MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507

python -m examples.solver_judge_tinker.train_solver_judge_flow_tinker \
    model.name=$MODEL_PATH \
    model.lora_rank=32 \
    training.group_size=4 \
    training.learning_rate=4e-5 \
    sampling.temperature=1.0 \
    sampling.top_p=1.0 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.grouping_level=trajectory \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.train_batch_size=64 \
    data.val_batch_size=512 \
    trainer.total_epochs=100 \
    trainer.logger=['wandb'] \
    trainer.project_name='solver-judge-workflow' \
    trainer.experiment_name='countdown-solver-judge-tinker-norm-by-std' \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=20 \
    trainer.default_local_dir='/tmp/countdown-solver-judge-tinker-norm-by-std'

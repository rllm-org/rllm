set -x

# On-policy distillation: student (Qwen3-8B) learns from teacher (Qwen3-235B-Instruct)
# 
# shared_tokenizer: Set to False for proper chat template handling, especially when:
#   - Student is a base model (Qwen3-8B) and teacher is an instruct model (Qwen3-235B-Instruct)
#   - The models may have different chat template expectations
# When False: Uses byte-level alignment and proper chat template parsing for teacher
# When True: Assumes tokenizers are identical and bypasses chat template parsing (faster but less robust)

python -m examples.solver_judge_distill.train_simple_math_distill_tinker \
    model.name=Qwen/Qwen3-8B \
    model.lora_rank=32 \
    training.group_size=1 \
    training.val_group_size=4 \
    training.learning_rate=4e-5 \
    sampling.temperature=1.0 \
    sampling.top_p=1.0 \
    algorithm.adv_estimator=distill \
    algorithm.shared_tokenizer=False \
    +algorithm.clip_advantages=True \
    +algorithm.adv_clip_min=-5.0 \
    +algorithm.adv_clip_max=5.0 \
    algorithm.teacher_rollout_args.backend=tinker \
    algorithm.teacher_rollout_args.model=Qwen/Qwen3-235B-A22B-Instruct-2507 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    trainer.total_epochs=100 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='solver-judge-distill' \
    trainer.experiment_name='simple-math-distill-8b-235b-cross-tokenizer' \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=20 \
    trainer.default_local_dir='./outputs/simple-math-distill-8b-235b-cross-tokenizer' \
    rollout_engine.bypass_render_with_parser=True \
    rollout_engine.disable_thinking=True \
    workflow.n_parallel_tasks=256 \
    training.rollout_timeout=1200


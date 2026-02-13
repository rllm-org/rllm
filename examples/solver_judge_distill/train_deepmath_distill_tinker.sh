set -x

python -m examples.solver_judge_distill.train_simple_math_distill_tinker \
    trainer.resume_from_tinker_id='tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/weights/final' \
    model.name=Qwen/Qwen3-8B-Base \
    model.lora_rank=128 \
    training.group_size=4 \
    training.val_group_size=4 \
    training.learning_rate=1e-4 \
    sampling.temperature=1.0 \
    sampling.top_p=1.0 \
    algorithm.adv_estimator=distill \
    algorithm.shared_tokenizer=True \
    +algorithm.clip_advantages=True \
    +algorithm.adv_clip_min=-5.0 \
    +algorithm.adv_clip_max=5.0 \
    algorithm.teacher_rollout_args.backend=tinker \
    algorithm.teacher_rollout_args.model=Qwen/Qwen3-32B \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    +data.max_response_length_val=32768 \
    +data.val_temperature=0.0 \
    data.train_batch_size=1024 \
    data.val_batch_size=32 \
    data.train_files='deepmath_opd:train' \
    data.val_files='deepmath_opd:test' \
    trainer.total_epochs=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='opd-deepmath-8b-32b' \
    trainer.experiment_name='deepmath-distill-8b-32b-rllm' \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.default_local_dir='./outputs/deepmath-distill-8b-32b-rllm' \
    rollout_engine.bypass_render_with_parser=False \
    rollout_engine.renderer_name=qwen3 \
    workflow.n_parallel_tasks=256 \
    training.rollout_timeout=1200

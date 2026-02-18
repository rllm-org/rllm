set -x

python -m examples.math_distill.opsd.train_deepmath_distill_tinker \
    trainer.resume_from_tinker_id='tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/weights/final' \
    model.name=Qwen/Qwen3-8B-Base \
    model.lora_rank=128 \
    training.group_size=4 \
    training.val_group_size=16 \
    training.learning_rate=1e-4 \
    sampling.temperature=1.0 \
    sampling.top_p=1.0 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    val_sampling.max_tokens=32768 \
    val_sampling.temperature=1.0 \
    data.train_batch_size=1024 \
    data.val_batch_size=512 \
    trainer.total_epochs=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='opsd-deepmath-8b' \
    trainer.experiment_name='opsd-deepmath-8b-rllm' \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.default_local_dir='./outputs/opsd-deepmath-8b-rllm' \
    rollout_engine.bypass_render_with_parser=False \
    rollout_engine.renderer_name=qwen3 \
    workflow.n_parallel_tasks=256

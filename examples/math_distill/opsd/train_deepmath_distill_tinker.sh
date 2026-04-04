set -x

python -m examples.math_distill.opsd.train_deepmath_distill_tinker \
    rllm/backend=tinker \
    training.resume_from_tinker_id='tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/weights/final' \
    model.name=Qwen/Qwen3-8B-Base \
    model.lora_rank=128 \
    training.group_size=4 \
    validation.group_size=8 \
    training.learning_rate=1e-4 \
    sampling.train.temperature=1.0 \
    sampling.val.temperature=1.0 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    +sampling.val.max_tokens=32768 \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger=['console','wandb'] \
    rllm.trainer.project_name='opd-deepmath-8b-32b' \
    rllm.trainer.experiment_name='opsd-deepmath-8b-rllm' \
    rllm.trainer.val_before_train=True \
    rllm.trainer.test_freq=10 \
    rllm.trainer.save_freq=10 \
    training.default_local_dir='./outputs/opsd-deepmath-8b-rllm' \
    rllm.algorithm.use_precomputed_advantage=true \
    rllm.algorithm.loss_fn=importance_sampling \
    rollout_engine.bypass_render_with_parser=True \
    rllm.workflow.n_parallel_tasks=512

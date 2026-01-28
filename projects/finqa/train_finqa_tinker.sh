#!/bin/bash
set -x

MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507

python3 -m projects.finqa.train_finqa_tinker \
    model.name=$MODEL_PATH \
    model.lora_rank=32 \
    training.group_size=16 \
    training.learning_rate=4e-5 \
    sampling.temperature=1.0 \
    sampling.top_p=1.0 \
    algorithm.adv_estimator=grpo \
    algorithm.grouping_level=trajectory \
    algorithm.norm_adv_by_std_in_grpo=false \
    data.max_prompt_length=4096 \
    data.max_response_length=12288 \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    workflow.n_parallel_tasks=4096 \
    trainer.total_epochs=10 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='finqa-tinker' \
    trainer.experiment_name='finqa-tool-tinker-v1' \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.default_local_dir='checkpoints/finqa-tinker'

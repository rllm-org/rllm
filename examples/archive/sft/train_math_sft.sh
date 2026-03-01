PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train_math_sft.py \
    model.partial_pretrain=Qwen/Qwen2.5-Math-7B-Instruct \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    trainer.total_epochs=3 \
    data.train_batch_size=4 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=13500 \
    data.truncation=right \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_files=train_data_5000.parquet \
    data.val_files=eval_data_1000.parquet \
    trainer.default_local_dir=outputs/qwen2.5_math_sft \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=math-tool-sft \
    optim.lr=1e-5

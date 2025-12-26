set -x


python3 -m examples.geo3k_tinker.train_geo3k_tinker \
  model.name=Qwen/Qwen3-VL-30B-A3B-Instruct \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  sampling.temperature=1.0 \
  sampling.top_p=1.0 \
  data.train_batch_size=32 \
  data.val_batch_size=64 \
  workflow.n_parallel_tasks=256 \
  trainer.total_epochs=10 \
  trainer.logger="['console','wandb']" \
  trainer.project_name='geo3k-tinker' \
  trainer.experiment_name='geo3k-tinker-full' \
  trainer.val_before_train=True


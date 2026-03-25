# 第一个 rLLM 训练（纯本地 GPU 版）：从 0 到跑通

这份教程只走一种路径：本地下载模型、本地启动推理服务、本地评测、本地训练。你不需要外部 API，也不需要执行 `rllm model setup`。

你将完成的闭环：

1. 安装 GPU 训练依赖（VERL 后端）
2. 下载模型到项目目录
3. 准备 GSM8K 数据
4. 启动本地 vLLM 推理服务
5. 做一次训练前 baseline 评测
6. 在本机 GPU 跑 LoRA + RL 训练
7. 训练后再次评测

---

## 1. 环境准备（GPU）

在仓库根目录执行：

```bash
cd /data/budget-tree/rllm

# 推荐 Python 3.11
uv venv --python 3.11
source .venv/bin/activate

# 安装 rLLM + verl（CUDA 12.8 示例）
uv pip install -e .[verl] --torch-backend=cu128

# 安装 HuggingFace CLI（用于下载模型）
uv pip install "huggingface_hub[cli]"

# 如需记录训练过程到 W&B，先设置 API key
export WANDB_API_KEY=your_wandb_api_key
```

如果你的 CUDA 版本不是 12.8，请把 `--torch-backend=cu128` 改成你机器对应版本。

如果你不想接 W&B，可以临时删除后面训练命令里的 `trainer.logger=['console','wandb']`，改回 `trainer.logger=['console']`。

---

## 2. 下载模型到项目目录

虽然 HuggingFace 模型可以在第一次运行时自动下载，但首次训练建议显式下载，避免训练启动后才发现网络或权限问题。

```bash
cd /data/budget-tree/rllm

mkdir -p models

# 如模型是 gated，先登录（可选）
huggingface-cli login

# 下载到项目目录
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
   --local-dir ./models/Qwen2.5-3B-Instruct
```

后面训练和评测统一使用这个路径：

`./models/Qwen2.5-3B-Instruct`

---

## 3. 准备数据

```bash
cd /data/budget-tree/rllm
python examples/gsm8k_lora/prepare_gsm8k_data.py
```

这一步会下载并注册 GSM8K 的 train/test split。

---

## 4. 启动本地 vLLM 推理服务

`rllm eval` 如果不传 `--base-url`，会走默认 provider 配置。为了保证你只执行教程里的命令就能跑通，这里显式启动一个本地 OpenAI 兼容服务。

在**另一个终端**里执行：

```bash
cd /data/budget-tree/rllm
source .venv/bin/activate

python -m vllm.entrypoints.openai.api_server \
   --model ./models/Qwen2.5-3B-Instruct \
   --host 127.0.0.1 \
   --port 8000 \
   --dtype bfloat16
```

保持这个终端不要关闭。后面的评测命令会访问 `http://127.0.0.1:8000/v1`。

如果你有多张 GPU，建议显式把评测服务固定在一张卡上。例如固定到 `GPU 1`：

```bash
cd /data/budget-tree/rllm
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
   --model ./models/Qwen2.5-3B-Instruct \
   --host 127.0.0.1 \
   --port 8000 \
   --dtype bfloat16
```

如果你只有一张 GPU，做完 baseline 评测后，先把这个 vLLM 服务停掉，再开始训练。训练结束后再重新启动它，做训练后评测。

---

## 5. 训练前先做 baseline

在主终端执行：

```bash
cd /data/budget-tree/rllm
rllm eval gsm8k \
   --base-url http://127.0.0.1:8000/v1 \
   --model ./models/Qwen2.5-3B-Instruct \
   --concurrency 8 \
   --max-examples 100
```

记下准确率等核心指标，后面用于对比。

---

## 6. 在本机 GPU 开始训练

先从一个更保守、更容易跑通的命令开始。

如果你只有一张 GPU，先停掉上一步的 vLLM 评测服务，再执行下面训练命令。

如果你有多张 GPU，建议显式把训练固定到另一张卡上。例如只让训练使用 `GPU 0`：

```bash
export CUDA_VISIBLE_DEVICES=0
```

```bash
cd /data/budget-tree/rllm
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

python3 examples/gsm8k_lora/train_gsm8k_with_lora.py \
   algorithm.adv_estimator=grpo \
   actor_rollout_ref.model.path=./models/Qwen2.5-3B-Instruct \
   trainer.n_gpus_per_node=1 \
   actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
   data.train_batch_size=1 \
   data.val_batch_size=32 \
   data.max_prompt_length=512 \
   data.max_response_length=1024 \
   actor_rollout_ref.model.lora_rank=32 \
   actor_rollout_ref.model.lora_alpha=32 \
   actor_rollout_ref.model.target_modules=all-linear \
   actor_rollout_ref.actor.optim.lr=5e-6 \
   actor_rollout_ref.actor.strategy=fsdp2 \
   actor_rollout_ref.actor.loss_agg_mode=token-mean \
   actor_rollout_ref.model.use_remove_padding=True \
   actor_rollout_ref.actor.ppo_mini_batch_size=1 \
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
   actor_rollout_ref.actor.use_dynamic_bsz=False \
   actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8000 \
   actor_rollout_ref.actor.use_kl_loss=False \
   actor_rollout_ref.actor.clip_ratio_high=0.2 \
   actor_rollout_ref.actor.kl_loss_coef=0.001 \
   actor_rollout_ref.actor.kl_loss_type=low_var_kl \
   actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
   actor_rollout_ref.model.enable_gradient_checkpointing=True \
   actor_rollout_ref.actor.fsdp_config.param_offload=False \
   actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
   actor_rollout_ref.rollout.n=2 \
   actor_rollout_ref.rollout.name=vllm \
   actor_rollout_ref.rollout.mode=async \
   actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
   actor_rollout_ref.rollout.enforce_eager=True \
   actor_rollout_ref.rollout.temperature=0.7 \
   actor_rollout_ref.rollout.top_p=0.95 \
   actor_rollout_ref.rollout.val_kwargs.n=1 \
   actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
   actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
   actor_rollout_ref.ref.fsdp_config.param_offload=False \
   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
   actor_rollout_ref.actor.entropy_coeff=0 \
   algorithm.kl_ctrl.kl_coef=0.001 \
   rllm.mask_truncated_samples=False \
   trainer.critic_warmup=0 \
   trainer.total_epochs=1 \
   trainer.project_name='rllm-experiment' \
   trainer.experiment_name='gsm8k-lora-local' \
   trainer.val_before_train=True \
   trainer.nnodes=1 \
   trainer.save_freq=1000 \
   trainer.test_freq=10 \
   trainer.default_hdfs_dir=null \
   rllm.agent.max_steps=1 \
   rllm.stepwise_advantage.enable=False \
   trainer.logger=['console','wandb']
```

这条命令的目标不是最快，而是优先提高“第一次就能启动成功”的概率。

---

## 7. 单卡 / 多卡如何改

你需要重点看这两个参数：

1. `trainer.n_gpus_per_node`
2. `actor_rollout_ref.rollout.tensor_model_parallel_size`

通常建议：

1. 单卡：
    `trainer.n_gpus_per_node=1`
    `actor_rollout_ref.rollout.tensor_model_parallel_size=1`
2. 双卡：
    `trainer.n_gpus_per_node=2`
    `actor_rollout_ref.rollout.tensor_model_parallel_size=2`
3. 四卡：
    `trainer.n_gpus_per_node=4`
    `actor_rollout_ref.rollout.tensor_model_parallel_size=2` 或 `4`

如果显存紧张，再优先下调：

1. `data.train_batch_size`
2. `actor_rollout_ref.rollout.n`
3. `actor_rollout_ref.actor.ppo_mini_batch_size`
4. `actor_rollout_ref.rollout.gpu_memory_utilization`

---

## 8. 训练后再评估

训练完成后，用同一套本地推理服务再跑一次评测：

```bash
cd /data/budget-tree/rllm
rllm eval gsm8k \
   --base-url http://127.0.0.1:8000/v1 \
   --model ./models/Qwen2.5-3B-Instruct \
   --concurrency 8 \
   --max-examples 100
```

对比训练前后指标：

1. 训练后提升：说明你的本机 GPU 首训闭环成功
2. 无提升或下降：优先调小学习率或延长训练步数

---

## 9. 最短可执行命令清单（GPU）

终端 A：先准备环境、模型和数据

```bash
cd /data/budget-tree/rllm
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .[verl] --torch-backend=cu128
uv pip install "huggingface_hub[cli]"
export WANDB_API_KEY=your_wandb_api_key

mkdir -p models
huggingface-cli login
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
   --local-dir ./models/Qwen2.5-3B-Instruct

python examples/gsm8k_lora/prepare_gsm8k_data.py
```

终端 B：启动本地推理服务

```bash
cd /data/budget-tree/rllm
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
   --model ./models/Qwen2.5-3B-Instruct \
   --host 127.0.0.1 \
   --port 8000 \
   --dtype bfloat16
```

如果你只有一张 GPU，把上面的 `CUDA_VISIBLE_DEVICES=1` 去掉，并且在训练前先手动停止这个 vLLM 服务。

终端 A：回到主流程，继续评测、训练、再评测

```bash
cd /data/budget-tree/rllm
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

rllm eval gsm8k \
   --base-url http://127.0.0.1:8000/v1 \
   --model ./models/Qwen2.5-3B-Instruct \
   --concurrency 8 \
   --max-examples 100

# 如果你只有一张 GPU，这里先停掉终端 B 里的 vLLM 服务，再继续训练

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

python3 examples/gsm8k_lora/train_gsm8k_with_lora.py \
   algorithm.adv_estimator=grpo \
   actor_rollout_ref.model.path=./models/Qwen2.5-3B-Instruct \
   trainer.n_gpus_per_node=1 \
   actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
   data.train_batch_size=1 \
   data.val_batch_size=32 \
   data.max_prompt_length=512 \
   data.max_response_length=1024 \
   actor_rollout_ref.model.lora_rank=32 \
   actor_rollout_ref.model.lora_alpha=32 \
   actor_rollout_ref.model.target_modules=all-linear \
   actor_rollout_ref.actor.optim.lr=5e-6 \
   actor_rollout_ref.actor.strategy=fsdp2 \
   actor_rollout_ref.actor.loss_agg_mode=token-mean \
   actor_rollout_ref.model.use_remove_padding=True \
   actor_rollout_ref.actor.ppo_mini_batch_size=1 \
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
   actor_rollout_ref.actor.use_dynamic_bsz=False \
   actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8000 \
   actor_rollout_ref.actor.use_kl_loss=False \
   actor_rollout_ref.actor.clip_ratio_high=0.2 \
   actor_rollout_ref.actor.kl_loss_coef=0.001 \
   actor_rollout_ref.actor.kl_loss_type=low_var_kl \
   actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
   actor_rollout_ref.model.enable_gradient_checkpointing=True \
   actor_rollout_ref.actor.fsdp_config.param_offload=False \
   actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
   actor_rollout_ref.rollout.n=2 \
   actor_rollout_ref.rollout.name=vllm \
   actor_rollout_ref.rollout.mode=async \
   actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
   actor_rollout_ref.rollout.enforce_eager=True \
   actor_rollout_ref.rollout.temperature=0.7 \
   actor_rollout_ref.rollout.top_p=0.95 \
   actor_rollout_ref.rollout.val_kwargs.n=1 \
   actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
   actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
   actor_rollout_ref.ref.fsdp_config.param_offload=False \
   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
   actor_rollout_ref.actor.entropy_coeff=0 \
   algorithm.kl_ctrl.kl_coef=0.001 \
   rllm.mask_truncated_samples=False \
   trainer.critic_warmup=0 \
   trainer.total_epochs=1 \
   trainer.project_name='rllm-experiment' \
   trainer.experiment_name='gsm8k-lora-local' \
   trainer.val_before_train=True \
   trainer.nnodes=1 \
   trainer.save_freq=1000 \
   trainer.test_freq=10 \
   trainer.default_hdfs_dir=null \
   rllm.agent.max_steps=1 \
   rllm.stepwise_advantage.enable=False \
   trainer.logger=['console','wandb']

# 如果你只有一张 GPU，训练结束后重新启动终端 B 里的 vLLM 服务，再做下面这次评测

rllm eval gsm8k \
   --base-url http://127.0.0.1:8000/v1 \
   --model ./models/Qwen2.5-3B-Instruct \
   --concurrency 8 \
   --max-examples 100
```

如果你按这个顺序执行，就能在不配置外部 provider 的前提下，完成一次纯本地 rLLM 训练闭环。

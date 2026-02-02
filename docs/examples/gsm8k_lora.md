# GSM8K LoRA Training

This example shows how to fine‑tune a math reasoning agent on the **GSM8K** dataset using **LoRA** in rLLM.  
You will use the standard `MathAgent` with a single‑turn environment and enable LoRA via a few configuration flags.

## Overview

With this example you will:

1. Prepare the GSM8K dataset and register it with `DatasetRegistry`
2. Train `MathAgent` on GSM8K using REINFORCE/GRPO‑style RL with LoRA adapters
3. Configure LoRA hyperparameters (rank, target modules, alpha) via the training script

The training loop uses the standard **VERL‑style backend** (`agent_ppo_trainer` config) and simply adds LoRA settings to the model.

---

## 1. Dataset Preparation

First, preprocess GSM8K and register it in rLLM:

```bash
cd examples/gsm8k_lora
python prepare_gsm8k_data.py
```

This will:

- Download the `openai/gsm8k` dataset (train + test)
- Extract the final numeric answer from the solution using a `#### <answer>` pattern
- Create a compact schema:
  - `question`: the original problem text
  - `ground_truth`: the extracted numeric answer
  - `data_source`: `"gsm8k"`
- Register datasets as:
  - `DatasetRegistry.load_dataset("gsm8k", "train")`
  - `DatasetRegistry.load_dataset("gsm8k", "test")`

Dataset preparation logic:

```python title="examples/gsm8k_lora/prepare_gsm8k_data.py"
--8<-- "examples/gsm8k_lora/prepare_gsm8k_data.py"
```

---

## 2. Training Script (LoRA + RL)

The main training entrypoint wraps `MathAgent` in a single‑turn environment with the built‑in math reward:

```python title="examples/gsm8k_lora/train_gsm8k_with_lora.py"
--8<-- "examples/gsm8k_lora/train_gsm8k_with_lora.py"
```

Key pieces:

- **Agent**: `MathAgent` from `rllm.agents.math_agent`
- **Environment**: `SingleTurnEnvironment` (one question → one answer)
- **Reward**: `math_reward_fn`, which parses the model’s output and checks correctness
- **Datasets**:
  - `train_dataset = DatasetRegistry.load_dataset("gsm8k", "train")`
  - `val_dataset = DatasetRegistry.load_dataset("gsm8k", "test")`

LoRA is configured via the **Hydra overrides** in the shell script rather than inside the Python file.

### 2.1 Launch training with LoRA

Use the helper shell script to start training with LoRA enabled:

```bash
cd examples/gsm8k_lora
bash train_gsm8k_lora.sh
```

Training configuration:

```bash title="examples/gsm8k_lora/train_gsm8k_lora.sh"
--8<-- "examples/gsm8k_lora/train_gsm8k_lora.sh"
```

Important LoRA‑related options in this script:

- `actor_rollout_ref.model.path=$MODEL_PATH` – base model (e.g. `Qwen/Qwen2.5-3B-Instruct`)
- `actor_rollout_ref.model.lora_rank=32` – LoRA rank
- `actor_rollout_ref.model.lora_alpha=32` – LoRA scaling
- `actor_rollout_ref.model.target_modules=all-linear` – apply LoRA to all linear layers

Other notable settings:

- `algorithm.adv_estimator=grpo` – GRPO‑style advantage estimation
- `data.max_prompt_length=512`, `data.max_response_length=1024`
- `trainer.logger=['console','wandb']`, `trainer.project_name='rllm-experiment'`, `trainer.experiment_name='gsm8k-lora'`

You can modify any of these via additional CLI overrides when calling the script.

---

## 3. Customizing LoRA and Training

To experiment with different LoRA and training parameters, you can directly override values in the script call, for example:

```bash
cd examples/gsm8k_lora
python3 -m examples.gsm8k_lora.train_gsm8k_lora \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=16 \
    data.train_batch_size=4 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    trainer.project_name='gsm8k-lora-ablation' \
    trainer.experiment_name='small-batch-lr-1e-5'
```

This GSM8K LoRA example demonstrates how **LoRA fine‑tuning is just a config change** on top of the standard rLLM RL training stack—no changes to the agent or environment code are required.



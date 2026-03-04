# RL Training with Tinker (using Unified Trainer)

This example shows how to train a **solver‑judge RL workflow** with the **Tinker** backend in rLLM, using Tinker's hosted GPU service and the **Unified Trainer**.

## Overview

With this example you will:

1. Train a **solver‑judge workflow** for the Countdown task using the Tinker backend
2. Use the **Unified Trainer** (`rllm.experimental.unified_trainer.AgentTrainer`) with the unified Hydra config system

Under the hood, rLLM integrates with Tinker as:

- **Rollout backend**: sampling and logprob computation happen on Tinker's GPU service
- **Policy trainer**: LoRA adapters are optimized remotely via Tinker training clients
- **Checkpoint manager**: checkpoints are stored and resumed via Tinker model IDs

---

## Setup

### Install dependencies

```bash
uv pip install -e .[tinker] --torch-backend=cpu
```

### Configure authentication

Set your API keys:

```bash
export TINKER_API_KEY=your_api_key_here
export WANDB_API_KEY=your_wandb_key_here  # optional, for W&B logging
```

You can obtain a Tinker API key from the Tinker console.

### Unified configuration system

This example uses the **unified Hydra config** introduced with the experimental unified trainer. Configuration is organized into:

- **Backend-agnostic rLLM configs** (`rllm/experimental/config/rllm/base.yaml`): core training settings shared across all backends
- **Tinker-specific configs** (`rllm/experimental/config/rllm/backend/tinker.yaml`): Tinker service, model, sampling, and data settings
- **Unified entry point** (`rllm/experimental/config/unified.yaml`): combines all configs

Key options you may want to tune:

| Config path | Description |
|---|---|
| `model.name` | Base model to fine‑tune (e.g. `Qwen/Qwen3-8B`) |
| `model.lora_rank` | LoRA rank |
| `training.group_size` | Number of trajectories per prompt (GRPO group size) |
| `data.max_prompt_length` / `data.max_response_length` | Context and generation lengths |
| `rllm.trainer.total_epochs` / `rllm.trainer.total_batches` | Training budget |
| `rllm.trainer.logger` | Logging backends (`console`, `wandb`, `tensorboard`) |
| `rllm.algorithm.adv_estimator` | Advantage estimator (`grpo`, `reinforce`, `rloo`, etc.) |

Backend-specific keys (under `model.*`, `training.*`, `sampling.*`, `data.*`) are forwarded into the common `rllm.*` namespace automatically. See the [rLLM and Backend Config](../experimental/rllm-and-backend-config.md) docs for details.

---

## Solver‑Judge RL Training with Tinker

This example trains a **multi‑agent solver‑judge workflow** on the Countdown task using the Tinker backend and the Unified Trainer.

### 1. Prepare Countdown dataset

First download and register the Countdown dataset:

```bash
cd examples/countdown
python prepare_countdown_data.py
```

This will:

- Load `Jiayi-Pan/Countdown-Tasks-3to4` from HuggingFace
- Convert each example into a math‑style word problem
- Register multiple splits (train, test, stage2, stage3) under the `countdown` key

Dataset preparation:

```python title="examples/countdown/prepare_countdown_data.py"
--8<-- "examples/countdown/prepare_countdown_data.py"
```

### 2. Training entrypoint

The training entrypoint uses `AgentTrainer` from the experimental unified trainer:

```python title="rllm/experimental/test_examples/test_tinker_solver_judge.py"
--8<-- "rllm/experimental/test_examples/test_tinker_solver_judge.py"
```

Key differences from the legacy trainer:

- Uses `rllm.experimental.unified_trainer.AgentTrainer` instead of `rllm.trainer.AgentTrainer`
- Config is loaded from `rllm.experimental.config` with `config_name="unified"`
- Supports per‑role advantage estimator mapping via `traj_group_adv_estimator_map` (e.g. GRPO for solver, REINFORCE for judge)

### 3. Train solver‑judge workflow with Tinker

Run training with the following script:

```bash
export TINKER_API_KEY=YOUR-TINKER-KEY
export WANDB_API_KEY=YOUR-WANDB-KEY # optional

MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
MODEL_LORA_RANK=32

# structure runname as <model_path>_<lora_rank>_<date>_<time>
date_str=$(date +%Y-%m-%d)
time_str=$(date +%H-%M-%S)
run_name=TINKER_${MODEL_PATH}_${MODEL_LORA_RANK}_${date_str}_${time_str}
local_dir=/path/to/your/local/dir

python3 -m rllm.experimental.test_examples.test_tinker_solver_judge \
    rllm/backend=tinker \
    rllm.compact_filtering.enable=False \
    rllm.algorithm.adv_estimator=grpo \
    rllm.algorithm.use_rllm=true \
    rllm.algorithm.norm_adv_by_std_in_grpo=true \
    rllm.trainer.total_batches=100 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.logger=['wandb'] \
    rllm.trainer.project_name='tinker-solver-judge' \
    rllm.trainer.experiment_name=$run_name \
    rllm.trainer.val_before_train=true \
    rllm.trainer.test_freq=20 \
    rllm.trainer.save_freq=20 \
    model.name=$MODEL_PATH \
    model.lora_rank=$MODEL_LORA_RANK \
    training.group_size=5 \
    training.learning_rate=4e-5 \
    training.default_local_dir=$local_dir \
    sampling.train.temperature=1.0 \
    sampling.train.top_p=1.0 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.train_batch_size=32 \
    data.val_batch_size=512
```

This will:

- Fine‑tune `Qwen/Qwen3-4B-Instruct-2507` with LoRA (rank 32)
- Use rLLM‑native advantage computation with per‑role estimators (GRPO for solver, REINFORCE for judge)
- Normalize advantages by standard deviation for stability
- Run validation before training and every 20 batches
- Log training metrics to Weights & Biases

You can customize training via Hydra CLI overrides. Note that:

- **rLLM-level settings** use the `rllm.*` prefix (e.g. `rllm.algorithm.adv_estimator`, `rllm.trainer.total_batches`)
- **Tinker backend settings** use their native keys (e.g. `model.name`, `training.group_size`, `sampling.train.temperature`)

For example, to switch to a larger model with a smaller LoRA rank:

```bash
python3 -m rllm.experimental.test_examples.test_tinker_solver_judge \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-8B \
    model.lora_rank=16 \
    training.group_size=8 \
    data.train_batch_size=32 \
    rllm.trainer.total_batches=200 \
    rllm.trainer.logger=['console','wandb'] \
    rllm.trainer.project_name='solver-judge-tinker' \
    rllm.trainer.experiment_name='countdown-grpo-qwen3-8b'
```

### 4. Run the workflow with Tinker rollout engine (optional)

For interactive evaluation (no training step), you can run the Countdown solver‑judge workflow directly using Tinker for sampling:

```python title="examples/solver_judge_tinker/run_solver_judge_flow_tinker.py"
--8<-- "examples/solver_judge_tinker/run_solver_judge_flow_tinker.py"
```

This script:

- Builds a `TinkerEngine` for rollouts
- Wraps it with `AgentWorkflowEngine` using `SolverJudgeWorkflow`
- Executes Countdown tasks and computes pass@1 / pass@k metrics

---

## Monitoring and Checkpoints

- **Logging**:
    - Set `rllm.trainer.logger=['console','wandb']` to enable Weights & Biases
    - Use `rllm.trainer.project_name` / `rllm.trainer.experiment_name` to organize runs
- **Checkpoints**:
    - Local paths are controlled by `training.default_local_dir`
    - You can resume from a Tinker checkpoint via `training.resume_from_tinker_id='tinker://<uuid>/weights/<checkpoint_name>'`

This gives you an end‑to‑end RL training pipeline where **rollouts, gradients, and checkpoints all run on Tinker's managed GPU service**, while rLLM's unified trainer handles datasets, workflows, advantage computation, and training orchestration.

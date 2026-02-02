# RL Training with Tinker

This example shows how to train a **solver‑judge RL workflow** with the **Tinker** backend in rLLM, using Tinker's hosted GPU service.

## Overview

With this example you will:

1. Train a **solver‑judge workflow** for the Countdown task using the Tinker backend
2. Reuse the unified Tinker RL config (`tinker_rl_trainer.yaml`) for workflow training

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

### Configure Tinker authentication

Set your Tinker API key:

```bash
export TINKER_API_KEY=your_api_key_here
```

You can obtain an API key from the Tinker console.

### Shared Tinker RL config

This example uses the unified RL config in rLLM:

```python title="rllm/trainer/config/tinker_rl_trainer.yaml"
--8<-- "rllm/trainer/config/tinker_rl_trainer.yaml"
```

Key options you may want to tune:

- `model.name`: base model to fine‑tune (e.g. `Qwen/Qwen3-8B`, `Qwen/Qwen3-30B-A3B`)
- `model.lora_rank`: LoRA rank
- `training.group_size`: number of trajectories per prompt (GRPO group size)
- `data.max_prompt_length` / `data.max_response_length`: context and generation lengths
- `trainer.total_epochs`, `trainer.logger`, `trainer.project_name`, `trainer.experiment_name`

You can override any of these from the command line using Hydra syntax (see below).

---

## Solver‑Judge RL Training with Tinker

This example trains a **multi‑agent solver‑judge workflow** on the Countdown task using the same Tinker RL backend.

### 2.1 Prepare Countdown dataset

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

### 2.2 Solver‑Judge workflow with Tinker backend

The Tinker RL training entrypoint for the solver‑judge workflow is:

```python title="examples/solver_judge_tinker/train_solver_judge_flow_tinker.py"
--8<-- "examples/solver_judge_tinker/train_solver_judge_flow_tinker.py"
```

It uses:

- `SolverJudgeWorkflow` from `examples.solver_judge.solver_judge_flow`
- `countdown_reward_fn` as the reward function
- `AgentTrainer` with `backend="tinker"` and `workflow_class=SolverJudgeWorkflow`

### 2.3 Train solver‑judge workflow with Tinker

Run the provided shell script:

```bash
cd examples/solver_judge_tinker
bash train_solver_judge_flow_tinker.sh
```

This will:

- Fine‑tune `Qwen/Qwen3-4B-Instruct-2507` with LoRA (rank 32)
- Train with GRPO using **trajectory‑level grouping** (`algorithm.grouping_level=trajectory`)
- Use normalized advantages for stability (`algorithm.norm_adv_by_std_in_grpo=true`)
- Log training metrics to Weights & Biases

Shell configuration:

```bash title="examples/solver_judge_tinker/train_solver_judge_flow_tinker.sh"
--8<-- "examples/solver_judge_tinker/train_solver_judge_flow_tinker.sh"
```

You can customize training via CLI overrides, e.g.:

```bash
cd examples/solver_judge_tinker
python -m examples.solver_judge_tinker.train_solver_judge_flow_tinker \
    model.name=Qwen/Qwen3-8B \
    model.lora_rank=16 \
    training.group_size=8 \
    data.train_batch_size=32 \
    trainer.total_epochs=20 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='solver-judge-tinker' \
    trainer.experiment_name='countdown-grpo-qwen3-8b'
```

### 2.4 Run the workflow with Tinker rollout engine (optional)

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

For the solver‑judge example:

- **Logging**:
  - Set `trainer.logger=['console','wandb']` to enable Weights & Biases
  - Use `trainer.project_name` / `trainer.experiment_name` to organize runs
- **Checkpoints**:
  - Local paths are controlled by `trainer.default_local_dir`
  - You can resume from a Tinker checkpoint via `trainer.resume_from_tinker_id='tinker://<uuid>/weights/<checkpoint_name>'`

This gives you an end‑to‑end RL training pipeline where **rollouts, gradients, and checkpoints all run on Tinker's managed GPU service**, while rLLM handles datasets, workflows, and trainer orchestration.



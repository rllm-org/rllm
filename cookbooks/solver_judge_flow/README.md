# Solver-Judge Flow

A multi-agent flow for rLLM that trains a solver-judge system on the countdown task using the **AgentFlow protocol**.

## Overview

The solver-judge pattern uses two agents cooperatively:

1. **Solver** — generates N candidate solutions to a countdown problem in parallel
2. **Judge** — evaluates the candidates and selects the best one

Both agents use a plain `OpenAI` client pointed at `config.base_url`. During training, this URL points to the model gateway which transparently captures token IDs and logprobs for RL optimization. During eval, it points directly to the model provider. The agent code is identical in both cases.

## Architecture

```
AgentFlow.run(task, config)
  │
  ├── Solver (N parallel threads)
  │     └── OpenAI(base_url=config.base_url).chat.completions.create(...)
  │         → Trajectory(name="solver", steps=[Step(action=parsed_answer)])
  │
  └── Judge
        └── OpenAI(base_url=config.base_url).chat.completions.create(...)
            → Trajectory(name="judge", steps=[Step(action=selected_answer)])
  │
  └── Episode(trajectories=[solver_0, solver_1, ..., judge])
```

The evaluator scores each trajectory independently:
- Solver trajectories are scored by whether their answer is correct
- Judge trajectory is scored by whether the selected answer is correct
- GRPO computes advantages separately for the `solver` and `judge` trajectory groups

## Installation

```bash
# From the rllm repo root
uv pip install -e ".[tinker]"                    # rllm + tinker backend
uv pip install -e cookbooks/solver_judge_flow    # this cookbook
```

After installation, the agent and evaluator are discoverable by the CLI:

```bash
rllm agent list    # should show "solver_judge" as a plugin
```

## Dataset

Pull the countdown dataset (one-time):

```bash
rllm dataset pull countdown
```

## Training

### Tinker (single-machine)

```bash
bash cookbooks/solver_judge_flow/train_tinker.sh
```

Or directly via the Python API:

```bash
python cookbooks/solver_judge_flow/train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    training.group_size=4
```

### Verl (distributed GPU)

Requires verl extras and megatron:

```bash
uv pip install -e ".[verl]"
bash scripts/install_megatron.sh <cu128|cu129|...>
```

Then:

```bash
bash cookbooks/solver_judge_flow/train_verl.sh
```

## Eval

```bash
rllm eval countdown \
    --agent solver_judge \
    --evaluator solver_judge_countdown \
    --model Qwen/Qwen3-8B
```

## Files

| File | Description |
|------|-------------|
| `solver_judge_flow.py` | `SolverJudgeFlow` — AgentFlow implementation |
| `evaluator.py` | `SolverJudgeCountdownEvaluator` — per-trajectory reward scoring |
| `train.py` | Python API training script (Hydra config) |
| `train_tinker.sh` | Tinker backend — single-machine training |
| `train_verl.sh` | Verl backend — distributed multi-GPU training |
| `pyproject.toml` | Plugin metadata and entry points |
| `test.py` | Unit tests for parsing and evaluation |

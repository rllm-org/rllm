# Math Tool Agent

A multi-turn agent flow for rLLM that trains a math agent to solve arithmetic problems using a **calculator tool**, demonstrating the **AgentFlow protocol** with tool use.

## Overview

The agent solves competition math problems from [DeepScaleR-Preview](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) (train, ~40K problems blending AIME/AMC/Omni-MATH) and [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (test) by calling a calculator tool for each arithmetic step. These problems are harder than GSM8K and require multi-turn reasoning, making them a better test for multi-turn RL training. This cookbook serves two purposes:

1. **End-to-end system test** — a multi-turn tool-use example that exercises the full training loop
2. **Onboarding** — shows new users how to build a tool-calling agent with rLLM

## Architecture

```
AgentFlow.run(task, config)
  │
  └── Multi-turn loop (up to 5 turns)
        │
        ├── LLM call via OpenAI(base_url=config.base_url)
        │     Model outputs reasoning + <tool_call>...</tool_call>
        │
        ├── Parse tool call → execute calculator → inject result
        │
        └── Repeat until model outputs <answer>NUMBER</answer>
```

The evaluator checks the final `<answer>` against the ground truth via numeric comparison.

## Installation

```bash
# From the rllm repo root
uv pip install -e ".[tinker]"                              # rllm + tinker backend
uv pip install --no-deps -e cookbooks/math_tool_agent      # this cookbook
```

After installation, the agent and evaluator are discoverable by the CLI:

```bash
rllm agent list    # should show "math_tool_agent" as a plugin
```

## Dataset

Pull the datasets (one-time):

```bash
rllm dataset pull deepscaler_math   # ~40K competition math problems (AIME/AMC/Omni-MATH/STILL)
rllm dataset pull math500           # 500-problem test benchmark
```

## Training

### Tinker (single-machine)

```bash
bash cookbooks/math_tool_agent/train_tinker.sh
```

Or directly via the Python API:

```bash
python cookbooks/math_tool_agent/train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    training.group_size=8
```

### Verl (distributed GPU)

Requires verl extras and megatron:

```bash
uv pip install -e ".[verl]"
bash scripts/install_megatron.sh <cu128|cu129|...>
```

Then:

```bash
bash cookbooks/math_tool_agent/train_verl.sh
```

## Tests

```bash
pytest cookbooks/math_tool_agent/test.py -v
```

## Files

| File | Description |
|------|-------------|
| `math_tool_agent.py` | `math_tool_agent` — multi-turn AgentFlow with calculator tool |
| `math_tool_eval.py` | `math_tool_evaluator` — numeric answer comparison |
| `train.py` | Python API training script (Hydra config) |
| `train_tinker.sh` | Tinker backend — single-machine training |
| `train_verl.sh` | Verl backend — distributed multi-GPU training |
| `pyproject.toml` | Plugin metadata and entry points |
| `test.py` | Unit tests for calculator, parsing, and evaluation |

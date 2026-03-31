# Geo3K Flow

A VLM geometry problem solver for rLLM that trains on the [Geometry3K](https://huggingface.co/datasets/hiyouga/geometry3k) dataset using the **AgentFlow protocol**.

## Overview

A single-turn VLM agent that receives a geometry problem with a diagram image and produces a step-by-step solution with a boxed final answer. Uses a plain `OpenAI` client with multimodal content blocks (base64-encoded images).

During training, `config.base_url` points to the model gateway which transparently captures token IDs and logprobs. During eval, it points directly to the model provider. The agent code is identical in both cases.

## Architecture

```
AgentFlow.run(task, config)
  │
  └── Solver
        └── OpenAI(base_url=config.base_url).chat.completions.create(
                messages=[system_prompt, {images + question}]
            )
            → Trajectory(name="solver", steps=[Step(action=response)])
  │
  └── Episode(trajectories=[solver], artifacts={"answer": response})
```

The evaluator extracts `\boxed{}` from the response and grades it against the ground truth using symbolic math grading.

## Installation

```bash
# From the rllm repo root
uv pip install -e ".[tinker]"          # rllm + tinker backend
uv pip install -e cookbooks/geo3k      # this cookbook
```

After installation, the agent and evaluator are discoverable by the CLI:

```bash
rllm agent list    # should show "geo3k" as a plugin
```

## Dataset

Pull the Geometry3K dataset (one-time):

```bash
rllm dataset pull geo3k
```

## Training

### Option 1: rllm CLI

```bash
rllm train geo3k \
    --agent geo3k \
    --evaluator geo3k_math \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --lora-rank 32 \
    --group-size 8 \
    --epochs 3
```

### Option 2: Python API

```bash
python cookbooks/geo3k/train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-VL-30B-A3B-Instruct \
    model.lora_rank=32 \
    training.group_size=8
```

Or use the provided script (wraps train.py with defaults):

```bash
bash cookbooks/geo3k/train.sh
```

## Eval

```bash
rllm eval geo3k \
    --agent geo3k \
    --evaluator geo3k_math \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct
```

## Files

| File | Description |
|------|-------------|
| `geo3k_flow.py` | `Geo3KFlow` — AgentFlow implementation (VLM single-turn solver) |
| `evaluator.py` | `Geo3KEvaluator` — math answer grading with `\boxed{}` extraction |
| `train.py` | Python API training script (Hydra config) |
| `train.sh` | Shell wrapper — calls `train.py` with default overrides |
| `pyproject.toml` | Plugin metadata and entry points |
| `test.py` | Unit tests for image handling and evaluation |
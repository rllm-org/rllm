# FinQA Financial Agent

This project demonstrates training and running [FinQA](https://rllm-project.com/post.html?post=finqa.md), a financial question-answering agent fine-tuned from Qwen3-4B-Instruct-2507 using rLLM. The agent uses specialized tools (SQL queries, table lookup, calculators) to perform multi-step reasoning over SEC 10-K financial statements, achieving **59.7%** on Snorkel Finance Benchmark — outperforming Qwen3-235B (51.4%) and rivaling Gemini 2.5 Pro (60.6%).

[Model Weights](https://huggingface.co/rLLM/rLLM-FinQA-4B) | [Dataset](https://huggingface.co/datasets/rLLM/finqa) | [Blog Post](https://rllm-project.com/post.html?post=finqa.md)

## Overview

The FinQA project demonstrates:

- How to use rLLM's ToolAgent and ToolEnvironment for multi-step financial reasoning
- How to build domain-specific tools in rLLM
- How to train agents with GRPO using LLM-as-judge rewards

## Quick Start

### Installation

Follow [rLLM installation](../getting-started/installation.md), then install FinQA dependencies:

```bash
uv pip install -r projects/finqa/requirements.txt
```

### Dataset Preparation

Downloads the [rLLM/finqa](https://huggingface.co/datasets/rLLM/finqa) dataset and registers train/val/test splits:

```bash
python -m projects.finqa.prepare_finqa_data
```

### Inference

Start a vLLM server and run the agent:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model rLLM/rLLM-FinQA-4B \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16

python -m projects.finqa.run_finqa
```

### Training

Set the required environment variables before training:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key for the reward judge |
| `PORTKEY_API_KEY` | Portkey gateway key for reward judge caching |

```bash
# verl backend (Qwen3-4B-Instruct-2507)
bash projects/finqa/train_finqa.sh

# tinker backend (Qwen3-30B-A3B-Instruct-2507, LoRA)
bash projects/finqa/train_finqa_tinker.sh
```

## Code Reference

### Financial Agent Runner

Main script for running financial reasoning:

```python title="projects/finqa/run_finqa.py"
--8<-- "projects/finqa/run_finqa.py"
```

### Training Script

FinQA training configuration:

```python title="projects/finqa/train_finqa.py"
--8<-- "projects/finqa/train_finqa.py"
```

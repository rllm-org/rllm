# FinQA Financial Agent

This example demonstrates training and running [FinQA](https://rllm-project.com/blog), a financial question-answering agent fine-tuned from Qwen3-4B-Instruct-2507 on SEC 10-K filings. The agent uses specialized tools (SQL queries, table lookup, calculators) to answer questions about financial statements, achieving 59.70% accuracy on Snorkel Finance Benchmark and 26.6% on Snorkel Finance Reasoning.

## Overview

The FinQA examples demonstrate:

- How to use rLLM's FinQA Agent for financial question-answering with tool use
- How to train agents with GRPO on multi-step tasks
- How to evaluate financial reasoning performance

## Quick Start

### Setup Financial Data

First, prepare your financial datasets:

```bash
cd projects/finqa
python prepare_finqa_data.py
```

### Model Hosting

Start a model server (choose one option):

**Option 1: Using vLLM**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```

**Option 2: Using SGLang**
```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --dp-size 1 \
    --dtype bfloat16
```

### Run FinQA Agent

Execute the financial reasoning agent for evaluation:

```bash
python run_finqa.py
```

### Train FinQA Agent

Train your own FinQA agent:

```bash
# Train with verl backend
bash train_finqa.sh

# Train with tinker backend
bash train_finqa_tinker.sh
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

For detailed setup instructions, see the [README](https://github.com/rllm-org/rllm/blob/main/projects/finqa/README.md) in the finqa project directory.

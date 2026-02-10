# FinQA Financial Agent

This project demonstrates training and running [FinQA](https://rllm-project.com/blog), a financial question-answering agent fine-tuned from Qwen3-4B-Instruct-2507 on SEC 10-K filings. The agent uses specialized tools (SQL queries, table lookup, calculators) to answer questions about financial statements, achieving **59.70%** on Snorkel Finance Benchmark and **26.6%** on Snorkel Finance Reasoning.

[Model Weights](https://huggingface.co/rLLM/rLLM-FinQA-4B) | [Dataset](https://huggingface.co/datasets/rLLM/finqa) | [Blog Post](https://rllm-project.com/blog)

## Overview

The FinQA project demonstrates:

- How to use rLLM's FinQA Agent for financial question-answering with tool use
- How to train agents with GRPO on multi-step tasks
- How to evaluate financial reasoning performance

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

### Model Hosting

Start a vLLM server with OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model rLLM/rLLM-FinQA-4B \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```

The server should be accessible at `http://localhost:30000/v1`

### Run FinQA Agent

Once your model server is running and datasets are prepared, run inference:

```bash
python -m projects.finqa.run_finqa
```

### Train FinQA Agent

Train your own FinQA agent:

```bash
# Train with verl backend
bash projects/finqa/train_finqa.sh

# Train with tinker backend
bash projects/finqa/train_finqa_tinker.sh
```

### Environment Variables

| Variable | Required For | Description |
|---|---|---|
| `FINQA_TABLES_ROOT` | Training | Path to company tables directory (default: `data/company_tables`) |
| `OPENAI_API_KEY` | Training | OpenAI API key for the reward judge |
| `PORTKEY_API_KEY` | Training | Portkey gateway key for reward judge caching |

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

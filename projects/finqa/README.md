# FinQA Training Example

This directory contains scripts for training and running FinQA, a financial question-answering agent that performs multi-step reasoning over SEC 10-K financial statements using specialized tools.

Our examples use the following:
* [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) as the base model
* [rLLM/finqa](https://huggingface.co/datasets/rLLM/finqa) dataset (5,110 Q&A pairs across 207 companies)
* gpt-5-nano reward judge with Portkey gateway for caching

[Model Weights](https://huggingface.co/rLLM/rLLM-FinQA-4B) | [Dataset](https://huggingface.co/datasets/rLLM/finqa) | [Blog Post](https://rllm-project.com/post.html?post=finqa.md)

## Agent Overview

The FinQA agent is a ReAct-style tool agent that answers financial questions by querying structured tables extracted from SEC 10-K filings. The agent has access to 4 specialized tools:

| Tool | Description |
|------|-------------|
| `get_table_names` | List available tables for a given company |
| `get_table_info` | Get table metadata, columns, dtypes, and sample values |
| `sql_query` | Execute SQL queries on in-memory SQLite tables |
| `calculator` | Evaluate mathematical expressions |

All table data is preloaded into in-memory SQLite for low latency runtime access.

## Installation

Follow [rLLM installation](../../docs/getting-started/installation.md), then install FinQA dependencies:

```bash
uv pip install -r projects/finqa/requirements.txt
```

## Dataset Preparation

Downloads the [rLLM/finqa](https://huggingface.co/datasets/rLLM/finqa) dataset and prepares it for training and evaluation:

```bash
python -m projects.finqa.prepare_finqa_data
```

This will:
- Download the dataset from HuggingFace ([rLLM/finqa](https://huggingface.co/datasets/rLLM/finqa))
- Extract company tables to `projects/finqa/data/company_tables/` (207 companies, 6,923 tables)
- Create train/val/test splits (4,030 / 522 / 558 examples)
- Register all splits with the rLLM DatasetRegistry

## Inference

Start a vLLM server and run the agent:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model rLLM/rLLM-FinQA-4B \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16

python -m projects.finqa.run_finqa
```

## Training

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

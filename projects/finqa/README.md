# FinQA Training Example

This directory contains examples for training and running FinQA, a financial question-answering agent fine-tuned from Qwen3-4B-Instruct-2507. The agent uses specialized tools (SQL queries, table/column lookup, calculators) to answer questions about SEC 10-K financial statements.

Our examples use the following:
* [rLLM/rLLM-FinQA-4B](https://huggingface.co/rLLM/rLLM-FinQA-4B) as the fine-tuned model (based on Qwen3-4B-Instruct-2507)
* [rLLM/finqa](https://huggingface.co/datasets/rLLM/finqa) dataset for training and evaluation

## Installation

Follow [rLLM installation](../../docs/getting-started/installation.md), then install FinQA dependencies:

```bash
# After installing rLLM and activating the venv
uv pip install -r projects/finqa/requirements.txt
```

## Model Hosting

Start a vLLM server with OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model rLLM/rLLM-FinQA-4B \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```

The server should be accessible at `http://localhost:30000/v1`

## Dataset Preparation

Downloads the [rLLM/finqa](https://huggingface.co/datasets/rLLM/finqa) dataset and registers train/val/test splits:

```bash
python -m projects.finqa.prepare_finqa_data
```

## Running Inference

Once your model server is running and datasets are prepared, you can run inference:

```bash
python -m projects.finqa.run_finqa
```

## Training

```bash
# verl backend
bash projects/finqa/train_finqa.sh

# Or train with tinker backend
bash projects/finqa/train_finqa_tinker.sh
```

### Environment Variables

| Variable | Required For | Description |
|---|---|---|
| `FINQA_TABLES_ROOT` | Training | Path to company tables directory (default: `data/company_tables`) |
| `OPENAI_API_KEY` | Training | OpenAI API key for the reward judge |
| `PORTKEY_API_KEY` | Training | Portkey gateway key for reward judge caching |

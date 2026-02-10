# FinQA Training Example

This directory contains examples for training and running FinQA, a financial question-answering agent fine-tuned from Qwen3-4B-Instruct-2507. The agent uses specialized tools (SQL queries, table/column lookup, calculators) to answer questions about SEC 10-K financial statements.

Our examples uses the following:
* Qwen/Qwen3-4B-Instruct-2507 as the base model
* [rLLM/finqa](https://huggingface.co/datasets/rLLM/finqa) dataset for training and evaluation

## Installation

Follow [rLLM installation](../../docs/getting-started/installation.md), then install FinQA dependencies:

```bash
# After installing rLLM and activating the venv
uv pip install -r projects/finqa/requirements.txt
```

## Model Hosting

### Option 1: Using vLLM

Start a vLLM server with OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```

### Option 2: Using SGLang

```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --dp-size 1 \
    --dtype bfloat16
# increase dp_size to enable data-parallel processing on multi-GPU
```

The server should be accessible at `http://localhost:30000/v1`

## Dataset Preparation

Prepare the FinQA dataset:

```bash
cd projects/finqa
python prepare_finqa_data.py
```

This will:
- Download the rLLM/finqa dataset from HuggingFace
- Register train/val/test splits with the RLLM DatasetRegistry

## Running Inference

Once your model server is running and datasets are prepared, you can run inference:

```bash
cd projects/finqa
python run_finqa.py
```

### Configuration Options

You can modify the inference script parameters:

- `--model`: Model to use (default: "Qwen/Qwen3-4B-Instruct-2507")
- `--port`: API server port (default: 30000)
- `--n`: Number of attempts per task for Pass@K evaluation (default: 4)
- `--output`: JSON file path to save results

The script will:
1. Load the FinQA test dataset
2. Repeat each problem n times for Pass@K evaluation
3. Run parallel and async trajectory collection using the agent execution engine
4. Evaluate results and report Pass@1 and Pass@K accuracy

## Training

### Basic Training

To train FinQA with GRPO:

```bash
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

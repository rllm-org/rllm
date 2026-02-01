# Math Tool Agent Examples

This directory contains examples for training and running math reasoning agents with tool usage capabilities using the RLLM framework. The math tool agent has access to a Python interepreter to solve mathematical problems through step-by-step reasoning and tool-use.

Our examples uses the following:
* Qwen3-4B as the base model
* DeepScaleR-Math dataset for training
* AIME2024 dataset for evaluation


## Model Hosting

### Option 1: Using vLLM

Start a vLLM server with OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 \
    --tensor-parallel-size 1
# For multi-GPU: set --tensor-parallel-size to the number of GPUs (e.g., 4 for 4 GPUs)
```

### Option 2: Using SGLang

```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-4B \ 
    --dp-size 1 \
    --dtype bfloat16
# increase dp_size to enable data-parallel processing on multi-GPU 
```

The server should be accessible at `http://localhost:30000/v1`

## Dataset Preparation

Prepare the required datasets (AIME 2024 for testing, DeepScaleR for training):

```bash
cd examples/math_tool
python prepare_math_data.py
```

This will:
- Download AIME 2024 dataset from HuggingFace
- Download DeepScaleR math dataset for training
- Register both datasets with the RLLM DatasetRegistry

## Running Inference

Once your model server is running and datasets are prepared, you can run inference:

```bash
cd examples/math_tool
python run_math_with_tool.py
```

### Configuration Options

You can modify the inference script parameters:

- `n_parallel_agents`: Number of parallel agents (default: 64)
- `model_name`: Model to use (default: "Qwen/Qwen3-4B")
- `base_url`: API server URL (default: "http://localhost:30000/v1")
- `max_response_length`: Maximum response length (default: 16384)
- `max_prompt_length`: Maximum prompt length (default: 2048)
- `temperature`: Sampling temperature (default: 0.6)
- `top_p`: Top-p sampling (default: 0.95)

The script will:
1. Load the AIME 2024 test dataset
2. Repeat each problem 4 times for Pass@K evaluation
3. Run parallel inference using the async agent execution engine
4. Evaluate results and report Pass@1 and Pass@K accuracy

## Training

### Basic Training

To train a math reasoning agent with tool usage:

```bash
bash examples/math_tool/train_math_with_tool.sh
```
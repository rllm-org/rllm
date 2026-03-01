# MiniWoB Agent Examples

This directory contains examples for training and running a simple web agent for the BrowserGym-MiniWoB++ environment using the RLLM framework. 

Our examples uses the following:
* Qwen3-1.7B as the base model
* 96 BrowserGym-MiniWoB++ tasks for training
* Rest of 29 BrowserGym-MiniWoB++ tasks as dataset for evaluation


## Model Hosting

### Option 1: Using vLLM

Start a vLLM server with OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 
```

### Option 2: Using SGLang

```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-1.7B \ 
    --dp-size 1 \
    --dtype bfloat16
# increase dp_size to enable data-parallel processing on multi-GPU 
```

The server should be accessible at `http://localhost:30000/v1`

## Environment Setup 

To setup the local environment, we first download BrowserGym

```bash
pip install browsergym
pip install playwright
playwright install chromium
```

Then we download the MiniWoB++ repository

```bash
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git
git -C "./miniwob-plusplus" reset --hard 7fd85d71a4b60325c6585396ec4f48377d049838
```

Before running the inference or training example, remember to update the script variable

```bash
export MINIWOB_URL="file://<PATH_TO_MINIWOB_PLUSPLUS_CLONED_REPO>/miniwob/html/miniwob/"
```

## Dataset Preparation

Prepare the required datasets:

```bash
cd examples/miniwob
python prepare_miniwob_data.py
```

This will:
- Partition the available 125 MiniWoB++ tasks into training and evaluation
- Register both datasets with the RLLM DatasetRegistry

## Running Inference

Once your model server is running and datasets are prepared, you can run inference:

```bash
cd examples/miniwob
python run_miniwob.py
```

The script will:
1. Load the MiniWoB++ test dataset
2. Run parallel inference using the async agent execution engine
3. Evaluate results and compute success rates

## Training

### Basic Training

To train a basic web agent:

```bash
bash examples/miniwob/train_miniwob.sh
```
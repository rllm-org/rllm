# Dense Search Training with RLLM

Train search agents using dense retrieval on Wikipedia with pre-built E5 embeddings. Only uses the dense index.

## Quick Setup

### 1. Download Data & Setup
```bash
cd examples/search
python download_search_data.py --data_dir ./search_data

cd examples/search/search_data/prebuilt_indices
cat part_aa part_ab > e5_Flat.index
```

Downloads:
- Wikipedia corpus from [PeterJinGo/wiki-18-corpus](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus)
- Pre-built E5 dense index from [PeterJinGo/wiki-18-e5-index](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index)

### 2. Launch Retrieval Server
```bash
bash retrieval/launch_server.sh ./search_data/prebuilt_indices 8000
```

## Model Hosting

### Option 1: Using vLLM

Start a vLLM server with OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85
```

### Option 2: Using SGLang

```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-4B \
    --dp-size 1 \
    --dtype bfloat16
```

The server should be accessible at `http://localhost:30000/v1`

### 3. Train Agent

Train the search agent using reinforcement learning:

```bash
export RETRIEVAL_SERVER_URL="http://127.0.0.1:8000"
bash train_search_agent.sh
```

This will:
- Fine-tune the Qwen3-4B model using PPO on HotpotQA dataset
- Use the local retrieval server for training
- Save checkpoints every 40 steps
- Run for 100 epochs with validation every 10 steps

### 4. Run/Evaluate Agent

Run evaluation with a trained model:

```bash
export RETRIEVAL_SERVER_URL="http://127.0.0.1:8000"
python run_search_agent.py
```


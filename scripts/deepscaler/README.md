
# DeepScaleR Scripts

We have created the `deepscaler` branch as a stable checkpoint to fully reproduce DeepScaleR training and evaluation.

## Single-Node Training: 8K Context

Our 8k context script runs on a single node with 8 A100-80GB GPUs:
```bash
# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
# Run 8K context length training
export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
./scripts/[deepscaler|deepcoder]/train/run_deepscaler_1.5b_8k.sh --model $MODEL_PATH
```

## Multi-Node Training (32 GPUs)

Our long-context runs (16K/24K) are distributed across 4 nodes with 8 A100-80GB GPUs each. To run, follow these steps:

1. On the head node:
```bash
# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
# Start Ray head node
ray start --head
```

2. On each worker node:
```bash
# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
# Connect to head node (replace with your head node's address)
ray start --address=[RAY_ADDRESS]
```

3. Finally, on the head node, run the training script:
```bash
# Run 16K or 24K context length training
./scripts/train/run_deepscaler_1.5b_[16k|24k].sh --model [CHECKPOINT_PATH]
```
We welcome the community to try out different models, context legnths, and RL parameters in the training scripts!

### Ablations

Finally, we provide ablations for the 2k/4k context runs in `scripts/ablation/`. To run:
```bash
./scripts/ablation/run_deepscaler_1.5b_[2k|4k].sh --model [CHECKPOINT_PATH]
```
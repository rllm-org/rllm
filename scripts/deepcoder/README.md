
# DeepCoder Scripts

Both 16K and 32K context runs require 32 GPUS (~2000 seconds per step). We recommend using A100-80GB, H100, or higher.

To run, follow these steps:

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
./scripts/deepcoder/train/deepcoder_14b_[16|32]k.sh --model [CHECKPOINT_PATH]
```
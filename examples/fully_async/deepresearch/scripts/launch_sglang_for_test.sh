#!/bin/bash

model_path=$1
model_name=${2:-Qwen/Qwen3-8B}
num_gpus=${3:-8}  # Default to 8 GPUs
base_port=30000


echo "==================================================="
echo "Launching SGLang servers on multiple GPUs"
echo "==================================================="
echo "Model path: ${model_path}"
echo "Model name: ${model_name}"
echo "Number of GPUs: ${num_gpus}"
echo "Base port: ${base_port}"
echo "==================================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Launch SGLang server on each GPU
for gpu_id in $(seq 0 $((num_gpus - 1))); do
    port=$((base_port + gpu_id + 1))

    echo "Starting SGLang server on GPU ${gpu_id}, port ${port}..."

    CUDA_VISIBLE_DEVICES=${gpu_id} nohup python -m sglang.launch_server \
        --model-path ${model_path} \
        --served-model-name ${model_name} \
        --port ${port} \
        --host 0.0.0.0 \
        --tensor-parallel-size 1 \
        --dtype float16 \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen \
        > logs/sglang_gpu${gpu_id}_port${port}.log 2>&1 &

    pid=$!
    echo "  ✓ GPU ${gpu_id} - PID: ${pid} - Port: ${port}"

    # Brief pause to avoid conflicts
    sleep 2
done

echo ""
echo "==================================================="
echo "All SGLang servers launched!"
echo "==================================================="
echo "Ports: ${base_port}+1 to $((base_port + num_gpus))"
echo ""
echo "To check status:"
echo "  ps aux | grep sglang"
echo "  tail -f logs/sglang_gpu*.log"
echo ""
echo "To kill all servers:"
echo "  pkill -f 'sglang.launch_server'"
echo "==================================================="


# Generate worker URLs for all running workers
worker_urls=""
for gpu_id in $(seq 0 $((num_gpus - 1))); do
    port=$((base_port + gpu_id + 1))
    if [ -z "$worker_urls" ]; then
        worker_urls="http://localhost:${port}"
    else
        worker_urls="${worker_urls} http://localhost:${port}"
    fi
done

echo "Launching SGLang router on port ${base_port}..."
echo "Worker URLs: ${worker_urls}"
echo ""

python -m sglang_router.launch_router \
    --worker-urls ${worker_urls} \
    --host 0.0.0.0 \
    --port 4000

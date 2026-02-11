#!/bin/bash

model_path=${1:-Qwen/Qwen3-8B}
model_name=${2:-Qwen/Qwen3-8B}
num_gpus=${3:-8}  # Default to 8 GPUs
base_port=30000

# Create logs directory if it doesn't exist
mkdir -p logs

# Launch vLLM server on each GPU
for gpu_id in $(seq 0 $((num_gpus - 1))); do
    port=$((base_port + gpu_id + 1))
    
    echo "Starting vLLM server on GPU ${gpu_id}, port ${port}..."
    
    CUDA_VISIBLE_DEVICES=${gpu_id} nohup vllm serve ${model_path} \
        --served-model-name ${model_name} \
        --port ${port} \
        --host 0.0.0.0 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.8 \
        > logs/vllm_gpu${gpu_id}_port${port}.log 2>&1 &
    
    pid=$!
    echo "  ✓ GPU ${gpu_id} - PID: ${pid} - Port: ${port}"
    
    # Brief pause to avoid conflicts
    sleep 2
done

echo ""
echo "==================================================="
echo "All vLLM servers launched!"
echo "==================================================="
echo "Ports: ${base_port}+1 to $((base_port + num_gpus))"
echo ""
echo "To check status:"
echo "  ps aux | grep vllm"
echo "  tail -f logs/vllm_gpu*.log"
echo ""
echo "To kill all servers:"
echo "  pkill -f 'vllm serve'"
echo "==================================================="
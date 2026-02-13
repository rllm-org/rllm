#!/bin/bash

# Usage: bash eval_one_ckpt.sh <ckpt_path> [model_name] [num_gpus]
# Example: bash eval_one_ckpt.sh /path/to/ckpts/deepresearch/8b_stale05_rs/global_step_200/merged_hf Qwen/Qwen3-8B 8

set -e  # Exit on error

ckpt_path=$1
model_name=${2:-Qwen/Qwen3-8B}
num_gpus=${3:-8}

if [ -z "$ckpt_path" ]; then
    echo "Error: checkpoint path is required"
    echo "Usage: bash eval_one_ckpt.sh <ckpt_path> [model_name] [num_gpus]"
    exit 1
fi

echo "==================================================="
echo "Evaluating checkpoint: ${ckpt_path}"
echo "Model name: ${model_name}"
echo "Number of GPUs: ${num_gpus}"
echo "==================================================="
echo ""

# Step 1: Activate conda environment
echo "[1/5] Activating conda environment: rllm-async"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rllm-async
echo "  ✓ Environment activated"
echo ""

# Step 2: Kill all existing VLLM and SGLang processes
echo "[2/5] Cleaning up existing processes..."

echo "  Killing SGLang processes..."
pkill -f "sglang.launch_server" || true
sleep 2
pkill -9 -f "sglang.launch_server" || true

echo "  Killing SGLang router processes..."
pkill -f "sglang_router.launch_router" || true
sleep 2
pkill -9 -f "sglang_router.launch_router" || true

echo "  Freeing Prometheus metrics port (29000)..."
# Kill any process using port 29000 (Prometheus metrics exporter)
if lsof -i :29000 > /dev/null 2>&1; then
    prometheus_pid=$(lsof -t -i :29000)
    if [ -n "$prometheus_pid" ]; then
        echo "    Found process on port 29000: PID ${prometheus_pid}"
        kill -9 $prometheus_pid || true
        sleep 1
        echo "    ✓ Port 29000 freed"
    fi
else
    echo "    ✓ Port 29000 already free"
fi

echo "  ✓ All processes cleaned up"
echo ""

# Step 3: Launch SGLang servers
echo "[3/5] Launching SGLang servers..."
cd /path/to/rllm
bash examples/fully_async/deepresearch/scripts/launch_sglang_for_test.sh \
    "${ckpt_path}" \
    "${model_name}" \
    "${num_gpus}" &

launcher_pid=$!
echo "  Launcher PID: ${launcher_pid}"
echo ""

# Step 4: Wait for all SGLang servers to start
echo "[4/5] Waiting for SGLang servers to start..."
base_port=30000
max_wait=300  # 5 minutes timeout
wait_interval=5

for gpu_id in $(seq 0 $((num_gpus - 1))); do
    port=$((base_port + gpu_id + 1))
    echo "  Waiting for server on port ${port}..."

    elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -s http://localhost:${port}/health > /dev/null 2>&1; then
            echo "    ✓ Server on port ${port} is ready"
            break
        fi
        sleep $wait_interval
        elapsed=$((elapsed + wait_interval))
    done

    if [ $elapsed -ge $max_wait ]; then
        echo "    ✗ Timeout waiting for server on port ${port}"
        echo "    Check logs: logs/sglang_gpu${gpu_id}_port${port}.log"
        exit 1
    fi
done

# Wait for router to start
echo "  Waiting for router on port 4000..."
elapsed=0
while [ $elapsed -lt $max_wait ]; do
    if curl -s http://localhost:4000/health > /dev/null 2>&1; then
        echo "    ✓ Router on port 4000 is ready"
        break
    fi
    sleep $wait_interval
    elapsed=$((elapsed + wait_interval))
done

if [ $elapsed -ge $max_wait ]; then
    echo "    ✗ Timeout waiting for router on port 4000"
    exit 1
fi

echo "  ✓ All servers are ready"
echo ""

# Step 5: Run evaluation
echo "[5/5] Running evaluation..."
timestamp=$(date +%Y%m%d_%H%M%S)
ckpt_name=$(basename $(dirname ${ckpt_path}))/$(basename ${ckpt_path})
log_file="eval_logs/eval_${ckpt_name//\//_}_${timestamp}.log"

mkdir -p eval_logs

echo "  Log file: ${log_file}"
echo "  Starting evaluation at $(date)"
echo ""

cd /path/to/rllm
python -m examples.fully_async.deepresearch.eval_browsecomp 2>&1 | tee "${log_file}"

echo ""
echo "==================================================="
echo "Evaluation completed!"
echo "==================================================="
echo "Log file: ${log_file}"
echo "Results saved to: eval_results_browsecomp_*.json"
echo "==================================================="

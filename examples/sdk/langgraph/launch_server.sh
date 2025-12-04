#!/bin/bash
#
# Launch script for the multi-GPU sharded retrieval server.
# Distributes the 60GB index across all GPUs (~7.5GB per GPU with 8 GPUs).
#
# Usage:
#     bash launch_server_gpu_sharded.sh [data_dir] [port] [ngpus] [device]
#
# Examples:
#     # Use all available GPUs for index, CPU for embedding (default)
#     bash launch_server_gpu_sharded.sh ./search_data/prebuilt_indices 9002
#
#     # Use only 4 GPUs for index, CPU for embedding
#     bash launch_server_gpu_sharded.sh ./search_data/prebuilt_indices 9002 4 cpu
#
#     # Use all GPUs for index, last GPU for embedding
#     bash launch_server_gpu_sharded.sh ./search_data/prebuilt_indices 9002 "" gpu
#

# Limit thread creation to prevent resource exhaustion
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Default values
DATA_DIR=${1:-"./search_data/prebuilt_indices"}
PORT=${2:-9002}
NGPUS=${3:-""}  # Empty means use all GPUs
DEVICE=${4:-"gpu"}  # cpu or gpu, defaults to cpu

echo "Starting Multi-GPU Sharded Retrieval Server..."
echo "Data directory: $DATA_DIR"
echo "Port: $PORT"
if [ -n "$NGPUS" ]; then
    echo "Number of GPUs for index: $NGPUS"
else
    echo "Number of GPUs for index: all available"
fi
echo "Embedding device: $DEVICE"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' not found!"
    exit 1
fi

# Check for required files
required_files=("corpus.json" "e5_Flat.index")
for file in "${required_files[@]}"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "Error: $file not found in $DATA_DIR"
        exit 1
    fi
done

# Check GPU availability and determine embedding device
EMBEDDING_GPU=""
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    echo ""
    
    # If device is GPU, use the last available GPU
    if [ "$DEVICE" = "gpu" ]; then
        TOTAL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
        EMBEDDING_GPU=$((TOTAL_GPUS - 1))
        echo "Embedding model will use GPU $EMBEDDING_GPU (last GPU)"
    else
        echo "Embedding model will use CPU"
    fi
else
    echo "Warning: nvidia-smi not found. Server will use CPU."
    DEVICE="cpu"
fi

# Start server
echo "Launching Multi-GPU Sharded Server..."
if [ -n "$NGPUS" ]; then
    if [ "$DEVICE" = "gpu" ]; then
        python rag_server.py --data_dir "$DATA_DIR" --port "$PORT" --ngpus "$NGPUS" --host 0.0.0.0 --embedding_device cuda --embedding_gpu "$EMBEDDING_GPU"
    else
        python rag_server.py --data_dir "$DATA_DIR" --port "$PORT" --ngpus "$NGPUS" --host 0.0.0.0 --embedding_device cpu
    fi
else
    if [ "$DEVICE" = "gpu" ]; then
        python rag_server.py --data_dir "$DATA_DIR" --port "$PORT" --host 0.0.0.0 --embedding_device cuda --embedding_gpu "$EMBEDDING_GPU"
    else
        python rag_server.py --data_dir "$DATA_DIR" --port "$PORT" --host 0.0.0.0 --embedding_device cpu
    fi
fi

echo "Server stopped."

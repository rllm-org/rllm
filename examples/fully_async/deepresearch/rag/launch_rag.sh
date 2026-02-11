#!/bin/bash
#
# Launch script for the multi-GPU sharded retrieval server v2 (AUTO-BATCHING).
#
# This version collects incoming requests and processes them in batches
# for significantly higher throughput. Both embedding and FAISS search are batched.
#
# Usage:
#     bash launch_rag.sh [data_dir] [port] [ngpus] [batch_timeout_ms] [max_batch_size] [faiss_temp_mem_mb] [faiss_use_float16] [faiss_query_batch_size]
#
# Examples:
#     # Default: 100ms batch timeout, max 64 queries per batch
#     bash launch_rag.sh ./search_data/prebuilt_indices 9002
#
#     # Faster batching: 50ms timeout, 128 max batch size
#     bash launch_rag.sh ./search_data/prebuilt_indices 9002 "" 50 128
#
#     # Lower latency: 20ms timeout, 32 max batch size
#     bash launch_rag.sh ./search_data/prebuilt_indices 9002 "" 20 32
#

# Limit thread creation to prevent resource exhaustion
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Default values
DATA_DIR=${1:-"/data/user/rllm/examples/sdk/langgraph/search_data/prebuilt_indices"}
PORT=${2:-9002}
NGPUS=${3:-""}  # Empty means use all GPUs for index
BATCH_TIMEOUT_MS=${4:-200}  # Batch timeout in milliseconds
MAX_BATCH_SIZE=${5:-256}  # Maximum queries per batch
FAISS_TEMP_MEM_MB=${6:-4096}  # FAISS per-GPU temp memory in MB
FAISS_USE_FLOAT16=${7:-0}  # 1 to enable float16 in FAISS GPU
FAISS_QUERY_BATCH_SIZE=${8:-16}  # Max queries per FAISS search() call (micro-batching)

# Convert ms to seconds for Python
BATCH_TIMEOUT=$(echo "scale=3; $BATCH_TIMEOUT_MS / 1000" | bc)

echo "=============================================="
echo "Multi-GPU RAG Server v2 (AUTO-BATCHING)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Port: $PORT"
echo "  Batch timeout: ${BATCH_TIMEOUT_MS}ms"
echo "  Max batch size: $MAX_BATCH_SIZE"
echo "  FAISS temp mem / GPU: ${FAISS_TEMP_MEM_MB}MB"
echo "  FAISS useFloat16: ${FAISS_USE_FLOAT16}"
echo "  FAISS query batch size: ${FAISS_QUERY_BATCH_SIZE}"
if [ -n "$NGPUS" ]; then
    echo "  GPUs for index: $NGPUS"
else
    echo "  GPUs for index: all available"
fi
echo ""

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

# Check GPU availability
EMBEDDING_GPU=0
if command -v nvidia-smi &> /dev/null; then
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    echo ""

    TOTAL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    echo "Total GPUs available: $TOTAL_GPUS"
    echo "Embedding model will use GPU 0"
    echo ""
else
    echo "Warning: nvidia-smi not found. Server will use CPU."
fi

# Start server
echo "=============================================="
echo "AUTO-BATCHING MODE"
echo "=============================================="
echo ""
echo "How it works:"
echo "  1. Requests are queued as they arrive"
echo "  2. Every ${BATCH_TIMEOUT_MS}ms OR when ${MAX_BATCH_SIZE} requests queue up:"
echo "     - All queries are encoded together (batch encoding)"
echo "     - All queries are searched together (batch FAISS search)"
echo "  3. Results are distributed back to each request"
echo ""
echo "Tuning tips:"
echo "  - Lower timeout = lower latency, but smaller batches"
echo "  - Higher timeout = higher throughput, but higher latency"
echo "  - Increase max_batch_size for high-concurrency workloads"
echo ""
echo "=============================================="
echo ""

# Launch the server
if [ -n "$NGPUS" ]; then
    python rag_server.py \
        --data_dir "$DATA_DIR" \
        --port "$PORT" \
        --ngpus "$NGPUS" \
        --host 0.0.0.0 \
        --embedding_device cuda \
        --embedding_gpu "$EMBEDDING_GPU" \
        --faiss_temp_mem_mb "$FAISS_TEMP_MEM_MB" \
        --faiss_query_batch_size "$FAISS_QUERY_BATCH_SIZE" \
        --batch_timeout "$BATCH_TIMEOUT" \
        --max_batch_size "$MAX_BATCH_SIZE" \
        $( [ "$FAISS_USE_FLOAT16" = "1" ] && echo "--faiss_use_float16" )
else
    python rag_server.py \
        --data_dir "$DATA_DIR" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --embedding_device cuda \
        --embedding_gpu "$EMBEDDING_GPU" \
        --faiss_temp_mem_mb "$FAISS_TEMP_MEM_MB" \
        --faiss_query_batch_size "$FAISS_QUERY_BATCH_SIZE" \
        --batch_timeout "$BATCH_TIMEOUT" \
        --max_batch_size "$MAX_BATCH_SIZE" \
        $( [ "$FAISS_USE_FLOAT16" = "1" ] && echo "--faiss_use_float16" )
fi

echo "Server stopped."
#!/bin/bash
#
# Launch script for the dense retrieval server (Search-R1 style).
#
# Environment setup:
#     conda create -n retriever python=3.10
#     conda activate retriever
#     conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
#     pip install transformers datasets pyserini
#     conda install -c pytorch -c nvidia faiss-gpu=1.8.0
#     pip install uvicorn fastapi
#
# Usage:
#     bash launch_server.sh [index_path] [corpus_path] [port]
#

# Default values
INDEX_PATH=${1:-"./search_data/prebuilt_indices/e5_Flat.index"}
CORPUS_PATH=${2:-"./search_data/wikipedia/wiki-18.jsonl"}
PORT=${3:-8000}

echo "Starting dense retrieval server (Search-R1 style)..."
echo "Index path: $INDEX_PATH"
echo "Corpus path: $CORPUS_PATH"
echo "Port: $PORT"

# Check if index file exists
if [ ! -f "$INDEX_PATH" ]; then
    echo "Error: Index file '$INDEX_PATH' not found!"
    echo "Please run download_search_data.py first:"
    echo "  python examples/search/download_search_data.py"
    exit 1
fi

# Check if corpus file exists
if [ ! -f "$CORPUS_PATH" ]; then
    echo "Error: Corpus file '$CORPUS_PATH' not found!"
    echo "Please run download_search_data.py first:"
    echo "  python examples/search/download_search_data.py"
    exit 1
fi

# Configuration
RETRIEVER_NAME="e5"
RETRIEVER_MODEL="intfloat/e5-base-v2"
TOPK=10
BATCH_SIZE=512
GPU_ID=0
LOG_LEVEL=${4:-"INFO"}  # Optional fourth argument: DEBUG, INFO, WARNING, ERROR

echo ""
echo "Configuration:"
echo "  Retriever: $RETRIEVER_NAME"
echo "  Model: $RETRIEVER_MODEL"
echo "  Top-K: $TOPK"
echo "  Batch size: $BATCH_SIZE"
echo "  GPU ID: $GPU_ID"
echo "  Log level: $LOG_LEVEL"
echo ""

# Start server
python retrieval/server.py \
    --index_path "$INDEX_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --topk $TOPK \
    --retriever_name "$RETRIEVER_NAME" \
    --retriever_model "$RETRIEVER_MODEL" \
    --faiss_gpu \
    --gpu_id $GPU_ID \
    --batch_size $BATCH_SIZE \
    --use_fp16 \
    --pooling_method mean \
    --host 0.0.0.0 \
    --port $PORT \
    --log_level "$LOG_LEVEL"

echo "Server stopped."

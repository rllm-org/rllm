#!/bin/bash
#
# Launch script for BrowseComp-Plus HTTP search server
#
# Usage:
#     bash launch_browsecomp.sh [searcher_type] [port]
#
# Examples:
#     bash launch_browsecomp.sh faiss 8000
#     bash launch_browsecomp.sh bm25 8000
#

set -e

SEARCHER_TYPE=${1:-"faiss"}
PORT=${2:-8000}
HOST=${3:-"0.0.0.0"}

# Get the script directory BEFORE changing directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTTP_SERVER="${SCRIPT_DIR}/browsecomp_http.py"

# Activate browsecomp conda environment and its venv
BROWSECOMP_DIR="/path/to/BrowseComp-Plus"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate browsecomp
source "${BROWSECOMP_DIR}/.venv/bin/activate"

# Index paths (BrowseComp-Plus)
BROWSECOMP_INDEX_DIR="${BROWSECOMP_DIR}/indexes"
BM25_INDEX_PATH="${BROWSECOMP_INDEX_DIR}/bm25/"
FAISS_INDEX_PATH="${BROWSECOMP_INDEX_DIR}/qwen3-embedding-8b/corpus.shard*.pkl"
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"

# Navigate to BrowseComp-Plus directory (for searcher imports)
cd "$BROWSECOMP_DIR"

# Add BrowseComp-Plus to PYTHONPATH for searcher module imports
export PYTHONPATH="${BROWSECOMP_DIR}:${PYTHONPATH}"

echo "=== BrowseComp-Plus HTTP Search Server ==="
echo "Searcher type: ${SEARCHER_TYPE}"
echo "Port: ${PORT}"
echo "Host: ${HOST}"

if [ "$SEARCHER_TYPE" = "faiss" ]; then
    echo "Index path: ${FAISS_INDEX_PATH}"
    echo "Model: ${EMBEDDING_MODEL}"
    echo ""

    python "${HTTP_SERVER}" --searcher-type faiss \
        --index-path "${FAISS_INDEX_PATH}" \
        --model-name "${EMBEDDING_MODEL}" \
        --normalize \
        --host "${HOST}" --port "${PORT}"

elif [ "$SEARCHER_TYPE" = "bm25" ]; then
    echo "Index path: ${BM25_INDEX_PATH}"
    echo ""

    python "${HTTP_SERVER}" --searcher-type bm25 \
        --index-path "${BM25_INDEX_PATH}" \
        --host "${HOST}" --port "${PORT}"

else
    echo "Error: Unknown searcher type '${SEARCHER_TYPE}'"
    echo "Supported types: faiss, bm25"
    exit 1
fi

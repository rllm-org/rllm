#!/bin/bash
# WebShop Data Setup Script
# This script downloads and prepares the data files required by WebShop environment.
#
# Usage:
#   ./setup_data.sh [-d small|all]
#
# Options:
#   -d small  Download only 1000 products (faster, recommended for testing)
#   -d all    Download entire dataset (1.18M products)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Displays information on how to use script
helpFunction() {
  echo "WebShop Data Setup Script"
  echo ""
  echo "Usage: $0 [-d small|all]"
  echo ""
  echo "Options:"
  echo "  -d small  Download 1000 products (faster, ~50MB, recommended for testing)"
  echo "  -d all    Download entire dataset (1.18M products, ~3GB)"
  echo ""
  echo "Example:"
  echo "  $0 -d small"
  exit 1
}

# Get values of command line flags
data=""
while getopts d:h flag
do
  case "${flag}" in
    d) data=${OPTARG};;
    h) helpFunction;;
  esac
done

if [ -z "$data" ]; then
  echo "[INFO]: No -d flag provided, defaulting to 'small' dataset"
  data="small"
fi

echo "=============================================="
echo "WebShop Data Setup"
echo "=============================================="
echo "Dataset size: $data"
echo "Working directory: $SCRIPT_DIR"
echo ""

# Set Java 11 for pyserini (requires Java 11+)
if [ -d "/usr/lib/jvm/java-11-openjdk-11.0.22.0.7-1.el7_9.x86_64" ]; then
  export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-11.0.22.0.7-1.el7_9.x86_64
  export PATH=$JAVA_HOME/bin:$PATH
  echo "[INFO] Using Java 11 from $JAVA_HOME"
elif [ -d "/usr/lib/jvm/java-11-openjdk" ]; then
  export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
  export PATH=$JAVA_HOME/bin:$PATH
  echo "[INFO] Using Java 11 from $JAVA_HOME"
elif [ -d "/usr/lib/jvm/java-11" ]; then
  export JAVA_HOME=/usr/lib/jvm/java-11
  export PATH=$JAVA_HOME/bin:$PATH
  echo "[INFO] Using Java 11 from $JAVA_HOME"
fi

# Verify Java version
java_version=$(java -version 2>&1 | head -n 1)
echo "[INFO] Java version: $java_version"

# Check if gdown is installed
if ! python3 -c "import gdown" &> /dev/null; then
  echo "[INFO] Installing gdown for Google Drive downloads..."
  pip3 install gdown
fi

# Create data directory
echo "[Step 1/4] Creating data directory..."
mkdir -p data
cd data

# Download product data
echo "[Step 2/4] Downloading product data from Google Drive..."
if [ "$data" == "small" ]; then
  if [ -f "items_shuffle_1000.json" ]; then
    echo "  - items_shuffle_1000.json already exists, skipping download."
  else
    echo "  - Downloading items_shuffle_1000.json (product info)..."
    gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib -O items_shuffle_1000.json --quiet
  fi
  if [ -f "items_ins_v2_1000.json" ]; then
    echo "  - items_ins_v2_1000.json already exists, skipping download."
  else
    echo "  - Downloading items_ins_v2_1000.json (product attributes)..."
    gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu -O items_ins_v2_1000.json --quiet
  fi
elif [ "$data" == "all" ]; then
  if [ -f "items_shuffle_1000.json" ]; then
    echo "  - items_shuffle_1000.json already exists, skipping download."
  else
    echo "  - Downloading items_shuffle_1000.json (product info - 1k subset)..."
    gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib -O items_shuffle_1000.json --quiet
  fi
  if [ -f "items_ins_v2_1000.json" ]; then
    echo "  - items_ins_v2_1000.json already exists, skipping download."
  else
    echo "  - Downloading items_ins_v2_1000.json (product attributes - 1k subset)..."
    gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu -O items_ins_v2_1000.json --quiet
  fi
  if [ -f "items_shuffle.json" ]; then
    echo "  - items_shuffle.json already exists, skipping download."
  else
    echo "  - Downloading items_shuffle.json (full product info)..."
    gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB -O items_shuffle.json --quiet
  fi
  if [ -f "items_ins_v2.json" ]; then
    echo "  - items_ins_v2.json already exists, skipping download."
  else
    echo "  - Downloading items_ins_v2.json (full product attributes)..."
    gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi -O items_ins_v2.json --quiet
  fi
else
  echo "[ERROR]: argument for '-d' flag not recognized: $data"
  helpFunction
fi

echo "  - Downloading items_human_ins.json (human instructions)..."
gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O -O items_human_ins.json --quiet

cd ..

# Download spaCy models
echo "[Step 3/4] Downloading spaCy language models..."
python3 -m spacy download en_core_web_lg --quiet 2>/dev/null || python3 -m spacy download en_core_web_lg
python3 -m spacy download en_core_web_sm --quiet 2>/dev/null || python3 -m spacy download en_core_web_sm

# Build search engine index
echo "[Step 4/4] Building search engine index..."
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
mkdir -p indexes indexes_100 indexes_1k indexes_100k

echo "  - Converting product data to search format..."
python3 convert_product_file_format.py

echo "  - Building Lucene indexes (this may take a while)..."
# Build indexes based on dataset size
if [ "$data" == "small" ]; then
  # Build main 'indexes' (used by default when num_products=None)
  echo "    Building indexes (main)..."
  python3 -m pyserini.index.lucene \
    --collection JsonCollection \
    --input resources \
    --index indexes \
    --generator DefaultLuceneDocumentGenerator \
    --threads 1 \
    --storePositions --storeDocvectors --storeRaw
  
  # Also build indexes_1k for explicit 1000 product setting
  echo "    Building indexes_1k..."
  python3 -m pyserini.index.lucene \
    --collection JsonCollection \
    --input resources_1k \
    --index indexes_1k \
    --generator DefaultLuceneDocumentGenerator \
    --threads 1 \
    --storePositions --storeDocvectors --storeRaw
else
  # Build all indexes for full dataset
  ./run_indexing.sh
fi

cd ..

echo ""
echo "=============================================="
echo "WebShop Data Setup Complete!"
echo "=============================================="
echo ""
echo "Data files downloaded to: $SCRIPT_DIR/data/"
ls -la data/
echo ""
echo "Search indexes built in: $SCRIPT_DIR/search_engine/indexes*/"
echo ""
echo "You can now run the WebShop tests:"
echo "  python3 -m pytest tests/envs/test_webshop_env.py"
echo ""

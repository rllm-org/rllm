#!/bin/bash
# Install Megatron-related dependencies for rLLM.
#
# These packages require special build flags that cannot be expressed
# in pyproject.toml. Run this AFTER installing the verl backend:
#   uv pip install -e ".[verl]" --torch-backend=<cu128|cu129|cu130|...>
#
# Usage:
#   bash scripts/install_megatron.sh <cu128|cu129|cu130|...>

set -euo pipefail

TORCH_BACKEND="${1:?Usage: bash scripts/install_megatron.sh <torch-backend, e.g. cu128 or cu129>}"
echo "=== Megatron dependency installer for rLLM ==="
echo "TORCH_BACKEND=${TORCH_BACKEND}"

echo "[1/5] Installing nvidia-modelopt..."
uv pip install 'nvidia-modelopt>=0.37.0'

echo "[2/5] Installing transformer-engine (this may take a while)..."
MAX_JOBS=128 uv pip install --no-cache --no-build-isolation "transformer_engine[pytorch]==2.10" --torch-backend="${TORCH_BACKEND}"

echo "[3/5] Installing megatron-core..."
uv pip install --no-deps "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.15.0"

echo "[4/5] Installing megatron-bridge..."
uv pip install --no-deps -U "git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@550924c04368a175ef261a72230204410f455260"

echo "[5/5] Installing NVIDIA Apex (required for gradient accumulation fusion)..."
APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
    uv pip install -v --no-cache --no-build-isolation \
    git+https://github.com/NVIDIA/apex.git --torch-backend="${TORCH_BACKEND}"

echo ""
echo "=== Megatron dependencies installed successfully ==="

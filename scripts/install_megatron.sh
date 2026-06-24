#!/bin/bash
# Install Megatron-related dependencies for rLLM.

set -euo pipefail

echo "=== Megatron dependency installer for rLLM ==="

echo "[1/5] Installing nvidia-modelopt..."
uv pip install 'nvidia-modelopt>=0.37.0'

echo "[2/5] Installing transformer-engine (this may take a while)..."
MAX_JOBS=128 uv pip install --no-cache --no-build-isolation "transformer_engine[pytorch]==2.11"

# megatron-core > 0.15.0 required for numpy>=2.0.0 compatibility
echo "[3/5] Installing megatron-core..."
uv pip install --no-deps megatron-core==0.17.1 

echo "[4/5] Installing megatron-bridge..."
# Pinned to 691a377f (2026-05-19): "Add external trainer integration helpers (#3813)"
# verl 0.8.0 requires LinearForLastLayer which was added in this commit (unreleased, post-v0.4.2).
uv pip install --no-deps git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@691a377f

echo "[5/5] Installing NVIDIA Apex (required for gradient accumulation fusion)..."
APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
    uv pip install -v --no-cache --no-build-isolation \
    git+https://github.com/NVIDIA/apex.git

echo ""
echo "=== Megatron dependencies installed successfully ==="

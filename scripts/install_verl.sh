#!/bin/bash
# Adapted from https://github.com/volcengine/verl/blob/v0.5.0/scripts/install_vllm_sglang_mcore.sh

echo "1. install inference frameworks and pytorch they need"
pip install "sglang[all]==0.4.6.post5" --no-cache-dir --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python && pip install torch-memory-saver --no-cache-dir
pip install --no-cache-dir "vllm==0.10.0" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata


echo "2. install basic packages"
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=19.0.1" pandas \
    "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler blobfile xgrammar \
    pytest py-spy pyext pre-commit ruff

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# Install flash-attn-2.8.0.post2 (cxx11abi=False)
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFalse-cp312-cp312-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFalse-cp312-cp312-linux_x86_64.whl

# Install flashinfer-0.2.2.post1+cu124 (cxx11abi=False)
# vllm-0.8.3 does not support flashinfer>=0.2.3
# see https://github.com/vllm-project/vllm/pull/15777
wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl && \
    pip install --no-cache-dir flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl


echo "4. May need to fix opencv"
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


echo "5. Install verl"
pip install --no-deps -e verl/
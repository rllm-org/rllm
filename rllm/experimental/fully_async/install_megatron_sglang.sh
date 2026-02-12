#!/bin/bash
# installing verl at commit adff7956cefd8ef707cd67dd8e08c06fa63679bd

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}

export MAX_JOBS=32

echo "1. install inference frameworks and pytorch they need"
if [ $USE_SGLANG -eq 1 ]; then
    uv pip install "sglang[all]==0.5.7" --no-cache && uv pip install torch-memory-saver --no-cache
fi
uv pip install --no-cache "vllm==0.15.0"
uv pip install sglang-router


echo "2. install basic packages"
uv pip install "transformers[hf_xet]>=4.51.0,<5.0.0" "accelerate" datasets peft hf-transfer \
    "numpy>=1.26,<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "uv pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# Install flash-attn-2.8.1 (cxx11abi=False)
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl && \
    uv pip install --no-cache flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

uv pip install --no-cache flashinfer-python==0.3.1


if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    uv pip install "onnxscript==0.3.1"
    NVTE_FRAMEWORK=pytorch uv pip install --no-cache --no-build-isolation --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.10
    uv pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.15.0
fi


echo "5. May need to fix opencv"
uv pip install opencv-python
uv pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid cudnn9.10+torch2.9.1 bug)"
    # Strip torch's pinned cudnn dep from metadata to avoid version conflicts
    TORCH_DIST_INFO=$(python -c "import torch, pathlib; print(next(pathlib.Path(torch.__file__).parent.parent.glob('torch-*.dist-info')))")
    if [ -f "${TORCH_DIST_INFO}/METADATA" ]; then
        sed -i '/nvidia-cudnn-cu12/d' "${TORCH_DIST_INFO}/METADATA"
    fi
    uv pip install --no-deps --force-reinstall nvidia-cudnn-cu12==9.16.0.29
fi

echo "Successfully installed all packages"

# install verl
# install rllm
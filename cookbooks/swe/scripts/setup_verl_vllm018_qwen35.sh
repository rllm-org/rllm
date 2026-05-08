#!/usr/bin/env bash
# Build the veRL + vLLM 0.18 environment used for Qwen3.5 SWE training.
#
# This script is intentionally explicit because Qwen3.5 support depends on a
# narrow stack: torch 2.10/cu129, vLLM 0.18, transformers 5.3,
# Megatron-Core 0.18, Megatron Bridge 0.4, flash-linear-attention, flash-attn,
# TransformerEngine, and Apex.
#
# Default output:
#   cookbooks/swe/.venv-verl-vllm018
#
# Typical use from the rLLM repo root:
#   bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
#
# Useful overrides:
#   VENV_ROOT=/path/to/venv-root bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
#   VENV_DIR=/path/to/.venv bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
#   VERL_PATH=/path/to/verl bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
#   MAX_JOBS=64 bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
#   RUN_SMOKE_TEST=0 bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh

set -euo pipefail

if [ "${TRACE:-0}" = "1" ]; then
    set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RLLM_REPO_ROOT="$(cd "$COOKBOOK_DIR/../.." && pwd)"

VENV_ROOT="${VENV_ROOT:-$COOKBOOK_DIR/.venv-verl-vllm018}"
VENV_DIR="${VENV_DIR:-$VENV_ROOT/.venv}"
VERL_PATH="${VERL_PATH:-}"
VERL_VERSION="${VERL_VERSION:-0.7.1}"
CACHE_DIR="${CACHE_DIR:-/tmp}"
MAX_JOBS="${MAX_JOBS:-32}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;10.0}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_DIR/uv_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_DIR/pip_cache}"
export TMPDIR="${TMPDIR:-$CACHE_DIR/build_tmp}"
mkdir -p "$VENV_ROOT" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

# Avoid importing packages from the devbox shell while building the venv.
unset PYTHONPATH

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
        export PATH="$CUDA_HOME/bin:$PATH"
    else
        echo "ERROR: nvcc was not found. Install or expose a CUDA toolkit before building flash-attn/TE." >&2
        exit 1
    fi
fi

echo "=== Qwen3.5 veRL/vLLM environment setup ==="
echo "rLLM root : $RLLM_REPO_ROOT"
echo "cookbook  : $COOKBOOK_DIR"
echo "venv      : $VENV_DIR"
echo "verl path : ${VERL_PATH:-<PyPI verl==$VERL_VERSION>}"
echo "uv        : $(uv --version)"
echo "nvcc      : $(nvcc --version | awk -F'release ' '/release/ {print $2}' | awk -F',' '{print $1}')"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu       : $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
fi
echo "arch list : $TORCH_CUDA_ARCH_LIST"
echo "max jobs  : $MAX_JOBS"
echo "==========================================="

uv python install 3.12
uv venv --python 3.12 "$VENV_DIR"

# Source-build packages call `python -m pip` through pyproject hooks.
cd "$CACHE_DIR"
uv pip install --python "$VENV_DIR/bin/python" pip

echo "[1/9] Installing torch 2.10.0 + CUDA 12.9 wheels"
uv pip install --python "$VENV_DIR/bin/python" \
    --index-url https://download.pytorch.org/whl/cu129 \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

echo "[2/9] Installing vLLM 0.18.0 and transformers 5.3.0"
uv pip install --python "$VENV_DIR/bin/python" vllm==0.18.0
uv pip install --python "$VENV_DIR/bin/python" \
    transformers==5.3.0 pybind11 ninja nvidia-mathdx

echo "[3/9] Installing Megatron-Core 0.18 and Megatron Bridge 0.4"
uv pip install --python "$VENV_DIR/bin/python" --no-deps \
    megatron-core==0.18.0 \
    megatron-bridge==0.4.0

echo "[4/9] Installing Python-level RL/runtime dependencies"
uv pip install --python "$VENV_DIR/bin/python" \
    flash-linear-attention==0.4.1 \
    peft==0.18.1 \
    trl==0.27.0 \
    liger-kernel \
    codetiming \
    mathruler \
    pylatexenc \
    qwen_vl_utils \
    cachetools \
    pytest \
    pytest-asyncio \
    tensordict \
    hydra-core \
    datasets \
    accelerate \
    "ray[default]" \
    wandb \
    nvtx \
    torchdata \
    modal \
    python-dotenv \
    litellm \
    pyyaml \
    pandas \
    openai \
    "mini-swe-agent[full]" \
    swebench \
    "swesmith[validate] @ git+https://github.com/SWE-bench/SWE-smith.git"

echo "[5/9] Preparing build environment for Apex, flash-attn, and TransformerEngine"
NVIDIA_SITE_PACKAGES="$VENV_DIR/lib/python3.12/site-packages/nvidia"
export NCCL_ROOT="$NVIDIA_SITE_PACKAGES/nccl"
export CPATH="$NCCL_ROOT/include:${CPATH:-}"
export LIBRARY_PATH="$NCCL_ROOT/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST MAX_JOBS

echo "[6/9] Building NVIDIA Apex"
"$VENV_DIR/bin/python" -m pip install -v --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex.git

echo "[7/9] Building flash-attn 2.8.3"
FLASH_ATTENTION_FORCE_BUILD=TRUE \
"$VENV_DIR/bin/python" -m pip install -v --no-build-isolation --no-cache-dir \
    flash_attn==2.8.3

echo "[8/9] Building TransformerEngine release_v2.12"
NVTE_FRAMEWORK=pytorch NVTE_BUILD_THREADS_PER_JOB=2 \
"$VENV_DIR/bin/python" -m pip install -v --no-build-isolation --no-cache-dir \
    "git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.12"

echo "[9/9] Installing veRL, rLLM, gateway, and SWE cookbook"
if [ -n "$VERL_PATH" ] && [ -d "$VERL_PATH" ]; then
    uv pip install --python "$VENV_DIR/bin/python" --no-deps -e "$VERL_PATH"
else
    if [ -n "$VERL_PATH" ]; then
        echo "WARNING: VERL_PATH does not exist: $VERL_PATH" >&2
    fi
    echo "Installing verl==$VERL_VERSION from PyPI. Set VERL_PATH for an editable checkout." >&2
    uv pip install --python "$VENV_DIR/bin/python" --no-deps "verl==$VERL_VERSION"
fi

uv pip install --python "$VENV_DIR/bin/python" --no-deps \
    -e "$RLLM_REPO_ROOT/rllm-model-gateway" \
    -e "$RLLM_REPO_ROOT" \
    -e "$COOKBOOK_DIR"

if [ "$RUN_SMOKE_TEST" = "1" ]; then
    echo "=== Running import/GPU smoke test ==="
    "$VENV_DIR/bin/python" - <<'PYEOF'
import torch
import vllm
import transformers
import megatron.core
import megatron.bridge
import flash_attn
import apex
import transformer_engine.pytorch as te
import verl
import rllm
import swe

print("--- versions ---")
print(f"torch              : {torch.__version__}")
print(f"cuda available     : {torch.cuda.is_available()}")
print(f"cuda devices       : {torch.cuda.device_count()}")
print(f"vllm               : {vllm.__version__}")
print(f"transformers       : {transformers.__version__}")
print("megatron.core      : OK")
print("megatron.bridge    : OK")
print(f"flash_attn         : {flash_attn.__version__}")
print("apex               : OK")
print("transformer_engine : OK")
print(f"verl               : {getattr(verl, '__version__', 'OK')}")
print(f"rllm               : {getattr(rllm, '__version__', 'OK')}")
print("swe cookbook       : OK")

if torch.cuda.is_available():
    from flash_attn import flash_attn_func
    q = torch.randn(2, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
    out = flash_attn_func(q, q, q)
    assert out.shape == q.shape
    lin = te.Linear(256, 256).cuda().bfloat16()
    x = torch.randn(4, 16, 256, device="cuda", dtype=torch.bfloat16)
    assert lin(x).shape == x.shape
    print("gpu smoke          : flash_attn_func + te.Linear bf16 OK")
else:
    print("gpu smoke          : skipped; CUDA is not available")
PYEOF
fi

cat <<NOTE

Setup complete.

Activate:
  source "$VENV_DIR/bin/activate"
  unset PYTHONPATH

Recommended runtime exports for training:
  export HF_HOME=\${HF_HOME:-/tmp/hf_cache}
  export TRITON_CACHE_DIR=\${TRITON_CACHE_DIR:-/tmp/triton_cache}
  export XDG_CACHE_HOME=\${XDG_CACHE_HOME:-/tmp/xdg_cache}
  export TORCH_EXTENSIONS_DIR=\${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}
  export CPATH="$NCCL_ROOT/include:\${CPATH:-}"
  export LD_LIBRARY_PATH="$NCCL_ROOT/lib:\${LD_LIBRARY_PATH:-}"

Then run:
  bash "$COOKBOOK_DIR/swe/training_scripts/run_swe_training_9b_megatron.sh"
  bash "$COOKBOOK_DIR/swe/training_scripts/run_swe_training_9b_megatron_h100.sh"

NOTE

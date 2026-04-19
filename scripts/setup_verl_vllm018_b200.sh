#!/usr/bin/env bash
# setup_verl_vllm018_b200.sh
#
# One-shot installer that reproduces the environment of
# `verlai/verl:vllm018.latest` as a native uv venv and installs rLLM on top.
# Produces a ~30 GB venv capable of running rLLM GRPO with the verl backend,
# Megatron-core, mbridge, and vLLM 0.18 on NVIDIA B200 (SM100 Blackwell).
# Expected wall time: 40–60 min on an 8×B200 devbox (bottlenecked by
# flash-attn + TransformerEngine source builds).
#
# This is an *alternative* to the two-step install documented in the rLLM
# README (`uv pip install -e ".[verl]" --torch-backend=cu129` followed by
# `bash scripts/install_megatron.sh cu129`). That path targets vLLM 0.17;
# this script targets vLLM 0.18 with Qwen3.5 + B200 support and must be
# run from a clean state (no prior `[verl]` extras installed).
#
# Verified on 2026-04-18 with:
#   host  : 8× NVIDIA B200 (SM10.0), driver 580.105.08, Debian 12
#   CUDA  : toolkit 12.9.86 at /usr/local/cuda
#   uv    : 0.9.8
#   python: 3.12.12 (installed by uv)
#   disk  : ~30 GB free needed under ${VENV_DIR%/.venv}, plus ~30 GB transient
#           in ${CACHE_DIR}
#
# Versions pinned to match verl's docker/Dockerfile.stable.vllm at main HEAD:
#   torch 2.10.0+cu129, vLLM 0.18.0, transformers 5.3.0,
#   Megatron-LM core_v0.16.0, mbridge @ 641a5a0, flash-attn 2.8.3,
#   TransformerEngine release_v2.12, flash-linear-attention 0.4.1,
#   peft 0.18.1, trl 0.27.0.
#
# Usage:
#   bash scripts/setup_verl_vllm018_b200.sh          # install from repo root
#   VENV_DIR=/path/to/.venv bash scripts/setup_verl_vllm018_b200.sh
#   MAX_JOBS=64 bash scripts/setup_verl_vllm018_b200.sh  # more parallel compile
#   VERL_PATH=/path/to/your/verl/fork bash scripts/setup_verl_vllm018_b200.sh
#
# Env overrides:
#   VENV_DIR   — where to create the venv (default: ./.venv at repo root)
#   VERL_PATH  — if set, install verl editable from this path. Otherwise,
#                install the pinned pip version matching rLLM's [verl] extra.
#   CACHE_DIR  — scratch dir for build objects and uv/pip caches
#                (default: /tmp)
#   MAX_JOBS   — parallelism for flash-attn/apex/TE (default: 32)
#   TORCH_CUDA_ARCH_LIST — GPU arches for CUDA compile (default "9.0;10.0")

set -xeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

############################# Config #############################
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
VERL_PATH="${VERL_PATH:-}"              # unset → install pinned wheel
VERL_PIN="${VERL_PIN:-verl==0.7.1}"     # only used when VERL_PATH is unset
CACHE_DIR="${CACHE_DIR:-/tmp}"
MAX_JOBS="${MAX_JOBS:-32}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;10.0}"

export UV_CACHE_DIR="${CACHE_DIR}/uv_cache"
export PIP_CACHE_DIR="${CACHE_DIR}/pip_cache"
export TMPDIR="${CACHE_DIR}/build_tmp"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

# Gotcha #3: some devboxes inject /mlx/workspace, /opt/tiger, ... into
# PYTHONPATH; leaving it set makes the new venv import host packages.
unset PYTHONPATH

############################# Pre-flight #########################
command -v uv   >/dev/null 2>&1 || { echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: CUDA toolkit (>=12.8) not found at /usr/local/cuda"; exit 1; }
NVCC_VER=$(nvcc --version | awk -F'release ' '/release/ {print $2}' | awk -F',' '{print $1}')
SM_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "pre-flight: nvcc ${NVCC_VER}, SM ${SM_CAP}, uv $(uv --version | awk '{print $2}')"
echo "pre-flight: repo root ${REPO_ROOT}"

############################# 1. Python 3.12 venv ################
uv python install 3.12
uv venv --python 3.12 "$VENV_DIR"

# Gotcha #1: `uv venv` does NOT install pip; source-build pkgs (flash-attn,
# apex, TE) call `python -m pip install ...` via pyproject-hooks and will
# fail with "No module named pip". Install pip explicitly.
# Gotcha #2: run subsequent `uv pip install` from a dir OUTSIDE any parent
# workspace that has its own pyproject.toml — otherwise uv may audit us
# against unrelated pins. The scratch dir avoids that.
cd "$CACHE_DIR"
uv pip install --python "$VENV_DIR/bin/python" pip

############################# 2. torch 2.10 + CUDA 12.9 ##########
uv pip install --python "$VENV_DIR/bin/python" \
    --index-url https://download.pytorch.org/whl/cu129 \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

############################# 3. vLLM + transformers #############
# vLLM 0.18 pulls transformers==4.57.6 transiently; we bump to 5.3.0 next
# to match Megatron-LM core_v0.16.0 expectations.
uv pip install --python "$VENV_DIR/bin/python" vllm==0.18.0
uv pip install --python "$VENV_DIR/bin/python" \
    transformers==5.3.0 pybind11 ninja nvidia-mathdx

############################# 4. Megatron + mbridge ##############
uv pip install --python "$VENV_DIR/bin/python" --no-deps \
    "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.16.0" \
    "git+https://github.com/ISEEKYAN/mbridge.git@641a5a0"

############################# 5. Pure-Python RL deps #############
# Gotcha #5: `torchdata` is imported by verl but not in its default install
# profile — added explicitly here.
uv pip install --python "$VENV_DIR/bin/python" \
    flash-linear-attention==0.4.1 peft==0.18.1 trl==0.27.0 \
    liger-kernel codetiming mathruler pylatexenc qwen_vl_utils \
    cachetools pytest-asyncio tensordict hydra-core datasets \
    accelerate "ray[default]" wandb nvtx torchdata

############################# 6. Heavy source builds #############
# Gotcha #4: TransformerEngine's CMake does NOT auto-find the NCCL headers
# shipped inside the venv's `nvidia-nccl-cu12` wheel. First attempt fails
# with `fatal error: nccl.h: No such file or directory`. Export NCCL_ROOT +
# CPATH + LIBRARY_PATH + LD_LIBRARY_PATH to the venv nccl dirs BEFORE the
# compile. These env vars are also useful at runtime; the training
# launcher must re-export them.
export NCCL_ROOT="$VENV_DIR/lib/python3.12/site-packages/nvidia/nccl"
export CPATH="$NCCL_ROOT/include:${CPATH:-}"
export LIBRARY_PATH="$NCCL_ROOT/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST MAX_JOBS

# 6a. Apex (~15 min). cpp_ext + cuda_ext; builds FusedAdam, FusedLayerNorm,
# multi_tensor_applier. Compiled for SM from TORCH_CUDA_ARCH_LIST.
"$VENV_DIR/bin/python" -m pip install -v --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex.git

# 6b. flash-attn 2.8.3 (~20 min). Forces source build because no cp312 +
# torch2.10 wheel exists on PyPI yet.
FLASH_ATTENTION_FORCE_BUILD=TRUE \
"$VENV_DIR/bin/python" -m pip install -v --no-build-isolation --no-cache-dir \
    flash_attn==2.8.3

# 6c. TransformerEngine release_v2.12 (~10 min with MAX_JOBS=64). Builds
# libtransformer_engine + TE-PyTorch extensions with FlashAttn-style fused
# attention and FP8/MXFP8/NVFP4 storage kernels.
NVTE_FRAMEWORK=pytorch NVTE_BUILD_THREADS_PER_JOB=2 \
"$VENV_DIR/bin/python" -m pip install -v --no-build-isolation --no-cache-dir \
    "git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.12"

############################# 7. verl + rLLM ####################
# verl: editable from VERL_PATH if provided, else pinned pip wheel.
if [[ -n "$VERL_PATH" ]]; then
    echo "Installing verl editable from $VERL_PATH"
    uv pip install --python "$VENV_DIR/bin/python" --no-deps -e "$VERL_PATH"
else
    echo "Installing verl pinned wheel: $VERL_PIN"
    uv pip install --python "$VENV_DIR/bin/python" --no-deps "$VERL_PIN"
fi

# rLLM itself (editable from this repo). --no-deps because the dependency
# closure was pinned explicitly above; the `[verl]` extra's deps would
# fight the pins we just installed (vllm 0.17 vs 0.18, flash-attn 2.8.1
# vs 2.8.3, transformers <5 vs 5.3).
uv pip install --python "$VENV_DIR/bin/python" --no-deps -e "$REPO_ROOT"

############################# 8. Smoke test ######################
"$VENV_DIR/bin/python" - <<'PYEOF'
import torch, vllm, transformers, mbridge, megatron, flash_attn, apex
import transformer_engine.pytorch as te
import verl
import rllm
print("--- versions ---")
print(f"torch              : {torch.__version__}  cuda={torch.cuda.is_available()} ngpu={torch.cuda.device_count()} cap={torch.cuda.get_device_capability(0)}")
print(f"vllm               : {vllm.__version__}")
print(f"transformers       : {transformers.__version__}")
print(f"mbridge            : {mbridge.__version__}")
print(f"megatron-core      : {megatron.core.__version__}")
print(f"flash_attn         : {flash_attn.__version__}")
print(f"apex               : OK")
print(f"transformer_engine : OK")
print(f"verl               : {verl.__version__}")
print(f"rllm               : {rllm.__version__}")
# quick GPU op
from flash_attn import flash_attn_func
q = torch.randn(2, 8, 128, 64, dtype=torch.bfloat16, device='cuda')
out = flash_attn_func(q, q, q)
assert out.shape == q.shape
lin = te.Linear(256, 256).cuda().bfloat16()
x = torch.randn(4, 16, 256, device='cuda', dtype=torch.bfloat16)
assert lin(x).shape == x.shape
print("--- smoke: flash_attn_func + te.Linear on B200 bf16: OK ---")
PYEOF

set +x
cat <<NOTE

=======================================================================
Setup complete.
  venv      : $VENV_DIR
  activate  : source $VENV_DIR/bin/activate && unset PYTHONPATH

To launch GRPO training with the verl backend, pick a script under
\`scripts/train/\` (e.g. \`bash scripts/train/simple_math.sh\`) or the
\`examples/\` subtree for a specific experiment.

Runtime env required by GRPO launchers (set these before you run):
  unset PYTHONPATH                             # gotcha #3
  export NCCL_SOCKET_IFNAME=lo                 # gotcha #6 (single-node: loopback)
  unset NCCL_SOCKET_FAMILY                     # gotcha #6 (some hosts preset AF_INET6 on iface w/o IPv6)
  export NCCL_IB_DISABLE=1                     # gotcha #6 (no IB needed intra-node)
  export CPATH=\$VENV/lib/python3.12/site-packages/nvidia/nccl/include:\$CPATH
  export LD_LIBRARY_PATH=\$VENV/lib/python3.12/site-packages/nvidia/nccl/lib:\$LD_LIBRARY_PATH

For bit-exact reproduction, snapshot the venv with \`uv pip freeze\` after
this completes and check the result into source control alongside this
script.
=======================================================================

NOTE

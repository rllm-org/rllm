#!/usr/bin/env bash
# Build a venv that can run the rLLM miles backend, without Miles' container image.
#
#   bash scripts/setup_miles_env.sh [VENV_DIR] [MILES_DIR] [SGLANG_DIR]
#
# Defaults put everything under ~/miles-env. Idempotent: re-running skips finished steps.
# Validated 2026-08-22 on 8xH100, driver CUDA 12.8, glibc 2.35.
set -euo pipefail

VENV="${1:-$HOME/miles-env/venv}"
MILES="${2:-$HOME/miles}"
SGLANG="${3:-$HOME/miles-env/sglang-miles}"
RLLM="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pin these. Both are branch checkouts with no release tags, so an unpinned setup can
# silently change weight-update or generation behaviour between runs.
SGLANG_BRANCH=sglang-miles
TORCH_VERSION=2.11.0          # whatever sglang pins; only the CUDA build differs
TORCH_CUDA=cu128              # must match the *driver's* CUDA, not the newest available
KERNEL_CUDA=cu129             # sglang-kernel index; cu13 wheels need driver >= 580

command -v uv >/dev/null || { echo "need uv on PATH"; exit 1; }
[ -d "$MILES" ] || { echo "no Miles checkout at $MILES -- git clone https://github.com/radixark/miles.git"; exit 1; }

echo "==> venv at $VENV"
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python"

echo "==> miles requirements"
# nvidia-resiliency-ext ships only manylinux_2_39 wheels (glibc 2.39 / Ubuntu 24.04) with
# no sdist, so it cannot install on older glibc. Fault tolerance only; dropping it means
# --use-fault-tolerance is unavailable.
grep -v '^nvidia-resiliency-ext' "$MILES/requirements.txt" > /tmp/miles-reqs-$$.txt
uv pip install --python "$PY" -q --prerelease=allow -r /tmp/miles-reqs-$$.txt
rm -f /tmp/miles-reqs-$$.txt

echo "==> miles (editable, no deps)"
uv pip install --python "$PY" -q --no-deps -e "$MILES"

echo "==> sglang: the miles fork, not PyPI"
# Stock sglang serves generation but has no /begin_weight_update, so weight sync 404s.
[ -d "$SGLANG/.git" ] || git clone --depth 1 --branch "$SGLANG_BRANCH" --single-branch \
    https://github.com/sgl-project/sglang.git "$SGLANG"
# --no-build-isolation: the isolated build env tries to compile a Rust dependency.
uv pip install --python "$PY" -q --no-deps --no-build-isolation -e "$SGLANG/python"

echo "==> torch/vision/audio for the driver's CUDA ($TORCH_CUDA)"
uv pip install --python "$PY" -q \
    --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
    "torch==$TORCH_VERSION" torchvision torchaudio

echo "==> sglang compiled kernels for $KERNEL_CUDA"
uv pip install --python "$PY" -q --reinstall-package sglang-kernel \
    --index-url "https://docs.sglang.ai/whl/$KERNEL_CUDA/" sglang-kernel
# cu13-only, and DeepGEMM needs SM100+ anyway; its presence breaks sglang's import.
uv pip uninstall --python "$PY" -q sgl-deep-gemm 2>/dev/null || true

echo "==> megatron-core (the FSDP path uses Megatron's fused cross-entropy) + ninja"
uv pip install --python "$PY" -q megatron-core ninja

echo "==> flash-attn (required, not optional: see below) -- source build, ~20-40 min"
# Miles' FSDP path packs samples into one row with attention_mask=None and derives
# cu_seqlens from packed position_ids. Only a varlen-capable attention kernel honours
# those boundaries; sdpa/eager silently let samples attend across each other, which
# corrupts the logprobs and shows up as a huge train/inference gap. Miles' image ships a
# prebuilt wheel; from source this compiles.
if ! "$PY" -c "import flash_attn" 2>/dev/null; then
    MAX_JOBS="${MAX_JOBS:-16}" uv pip install --python "$PY" --no-build-isolation flash-attn
fi

echo "==> rLLM (editable)"
uv pip install --python "$PY" -q -e "$RLLM"

echo
echo "==> verifying"
"$PY" - <<'CHECK'
import importlib, torch
for m in ("miles.utils.types", "miles.utils.arguments",
          "miles.ray.rollout.train_data_conversion",
          "miles.backends.training_utils.data", "miles.ray.placement_group",
          "sglang.srt.server_args", "sgl_kernel", "megatron.core", "rllm"):
    importlib.import_module(m)
    print(f"  ok  {m}")
assert torch.cuda.is_available(), (
    f"torch {torch.__version__} cannot see the GPUs: its CUDA build must match the driver "
    "(nvidia-smi 'CUDA Version'), and cu13 wheels need driver >= 580"
)
print(f"  ok  torch {torch.__version__}, {torch.cuda.device_count()} GPUs")
CHECK
echo
echo "Done. Run training with:"
echo "  PYTHON=$PY bash examples/countdown/unified_trainer/train_countdown_unified_miles.sh"

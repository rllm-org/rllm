#!/usr/bin/env bash
# Setup Qwen 3.5 FSDP training with rllm
#
# Usage:
#   pip install -e ".[verl]"
#   bash scripts/setup_qwen35.sh
#
# Installs exact package versions tested with Qwen3.5-0.8B GRPO on B200 GPUs.
# Several packages have conflicting pins (vllm vs transformers, verl vs numpy)
# so they must be installed with --no-deps.
set -euo pipefail

echo "=== Installing Qwen 3.5 dependencies ==="

# Core packages (--no-deps to avoid conflicts between vllm<->transformers, verl<->numpy)
uv pip install --no-deps \
    "verl @ git+https://github.com/verl-project/verl.git@main" \
    "vllm==0.18.0" \
    "transformers==5.3.0" \
    "huggingface-hub==1.8.0" \
    "numpy==2.2.6"

# Install vllm's remaining deps (excluding the conflicting ones we already installed)
uv pip install --no-deps \
    "torchvision==0.25.0" \
    "torchaudio==2.10.0"

echo "=== Verifying ==="
python3 -c "
import verl; print(f'verl: {verl.__version__}')
import vllm; print(f'vllm: {vllm.__version__}')
import torch; print(f'torch: {torch.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True)
print(f'Qwen3.5 model_type: {cfg.model_type}')
print('Setup complete!')
"

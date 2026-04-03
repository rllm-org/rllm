#!/usr/bin/env bash
# Train SWE-bench agent via train.py with Hydra overrides.
#
# Prerequisites:
#   1. Install rllm with tinker extras:  uv pip install -e ".[tinker]"
#   2. Install this cookbook:             uv pip install -e cookbooks/swe
#   3. Register datasets:                python cookbooks/swe/prepare_tinker_data.py --dataset swe_smith_py
#   4. Modal setup:                      modal setup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -u "$SCRIPT_DIR/train.py" \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3.5-35B-A3B \
    model.lora_rank=8 \
    training.group_size=8 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=10 \
    rllm.trainer.project_name=rllm-swe \
    rllm.trainer.experiment_name=swe-bench-cookbook \
    "$@"

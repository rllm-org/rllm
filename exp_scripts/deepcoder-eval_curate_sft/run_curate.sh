#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$HERE/.." && pwd)"
DEEPCODER_RUN="$WORKSPACE/deepcoder-run"
source "$WORKSPACE/finqa-grpo-run/env.sh"
source "$VENV/bin/activate"

export RLLM_HOME="${RLLM_HOME:-$DEEPCODER_RUN/runtime}"
export PYTHONPATH="$DEEPCODER_RUN:/home/yanan/agents/rllm/cookbooks/deepcoder:/home/yanan/agents/rllm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RUN_DIR="${1:?usage: run_curate.sh EVAL_RUN_DIR [OUTPUT_DIR]}"
OUT="${2:-${RUN_DIR%/}-sft-data}"
MODEL="${MODEL:-/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
MAX_SELECT_ROLLOUTS_CNT="${MAX_SELECT_ROLLOUTS_CNT:-1}"

python -u "$HERE/curate_dataset.py" \
  --run-dir "$RUN_DIR" \
  --output "$OUT" \
  --model "$MODEL" \
  --max-length 32768 \
  --max-select-rollouts-cnt "$MAX_SELECT_ROLLOUTS_CNT" \
  --val-fraction 0 \
  --split-seed 1234

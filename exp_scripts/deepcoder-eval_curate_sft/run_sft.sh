#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$HERE/.." && pwd)"
DEEPCODER_RUN="$WORKSPACE/deepcoder-run"
source "$WORKSPACE/finqa-grpo-run/env.sh"
source "$VENV/bin/activate"

export RLLM_HOME="${RLLM_HOME:-$DEEPCODER_RUN/runtime}"
export PYTHONPATH="/home/yanan/agents/rllm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

DATA_DIR="${1:?usage: run_sft.sh CURATED_DATA_DIR [OUTPUT_DIR]}"
OUT="${2:-/home/yanan/.deepcoder-checkpoints/deepcoder-self-sft-qwen3-4b-r32}"
MODEL="${MODEL:-/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
GPUS="${GPUS:-2}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
SAVE_FREQ="${SAVE_FREQ:-20}"
VAL_FREQ="${VAL_FREQ:-20}"

test -s "$DATA_DIR/train.jsonl"
test -s "$DATA_DIR/curation_manifest.json"
if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite SFT output $OUT" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

args=(
  sft
  --train-file "$DATA_DIR/train.jsonl"
  --model "$MODEL"
  --backend verl
  --gpus "$GPUS"
  --lora-rank 32
  --lr 1e-5
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --max-length "$MAX_LENGTH"
  --tokenize-method hf_template
  --lr-schedule constant
  --save-freq "$SAVE_FREQ"
  --val-freq "$VAL_FREQ"
  --project deepcoder-self-sft
  --experiment qwen3-4b-k8-success
  --output "$OUT"
)
if [[ -s "$DATA_DIR/val.jsonl" ]]; then args+=(--val-file "$DATA_DIR/val.jsonl"); fi

python -m rllm.cli.main "${args[@]}"

#!/usr/bin/env bash
# sft_deepseek_flash — distill deepseek-v4-pro terminus2 traces into deepseek-v4-flash (Fireworks LoRA SFT).
#
# Pipeline (all vars baked in for reproducibility):
#   1. Build a THINKING-SFT dataset from an eval run — preserves per-turn reasoning
#      as trained <think> blocks (stock `rllm dataset from-eval` drops reasoning).
#   2. Launch Fireworks SFT with the deepseek-correct renderer + v4 tokenizer + shape.
#
# Why a custom script (not plain `rllm sft`): the CLI can't set renderer_name /
# tokenizer_model / training-shape, and its defaults (role_colon + qwen tokenizer)
# would silently corrupt a deepseek think-model. sft_deepseek_flash.py injects the
# correct values via SFTSpec.overrides.
#
# Before running:
#   export FIREWORKS_API_KEY=...
#
# Run (from anywhere; cd's itself):
#   bash cookbooks/terminal-rl/sft_deepseek_flash.sh <eval_run_id_or_dir>
# e.g.
#   bash cookbooks/terminal-rl/sft_deepseek_flash.sh tb-v2-opus-pass-at-16_accounts_fireworks_models_deepseek-v4-pro_20260709_175947

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

EVAL_RUN="${1:?usage: sft_deepseek_flash.sh <eval_run_id_or_dir>}"
DATASET_NAME="${DATASET_NAME:-tb2-dsv4-distill}"

# ---- knobs (override via env) ----
export EPOCHS="${EPOCHS:-1}"
export LR="${LR:-1e-5}"
export LORA_RANK="${LORA_RANK:-32}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export MAX_LENGTH="${MAX_LENGTH:-32768}"
export REPLICA_COUNT="${REPLICA_COUNT:-1}"

echo "== 1/2: building thinking-SFT dataset '${DATASET_NAME}' from ${EVAL_RUN} =="
python build_distill_dataset.py "${EVAL_RUN}" --name "${DATASET_NAME}"

echo "== 2/2: launching Fireworks SFT of deepseek-v4-flash =="
python sft_deepseek_flash.py "${DATASET_NAME}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --lora-rank "${LORA_RANK}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --replica-count "${REPLICA_COUNT}" \
    "${@:2}"

#!/usr/bin/env bash
# Formal GRPO wrapper for the .16 two-GPU host.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
export FINQA_MULTI_ALGORITHM=grpo
export FINQA_MULTI_TRAIN_BATCH_SIZE=8
export FINQA_MULTI_EPOCHS=50
export FINQA_DISABLE_NCCL_P2P=0
export FINQA_MULTI_RUN_HOST_TAG=16
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-finqa-multitable-v2-grpo-b8-16}"

exec "$RUN_DIR/train_16_multitable_grpo.sh" "${1:-smoke}" "${@:2}"

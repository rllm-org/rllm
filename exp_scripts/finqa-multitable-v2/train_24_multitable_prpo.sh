#!/usr/bin/env bash
# Formal/smoke synchronous PRPO wrapper for the .24 two-GPU host.
# All training mechanics remain in the shared experiment entrypoint and the
# unmodified rLLM/veRL stack.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
export FINQA_MULTI_ALGORITHM=prpo
export FINQA_MULTI_TRAIN_BATCH_SIZE=64
export FINQA_MULTI_EPOCHS=50
export FINQA_DISABLE_NCCL_P2P=1
export FINQA_MULTI_RUN_HOST_TAG=24
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-finqa-multitable-v2-prpo-b64-24}"

exec "$RUN_DIR/train_16_multitable_grpo.sh" "${1:-smoke}" "${@:2}"

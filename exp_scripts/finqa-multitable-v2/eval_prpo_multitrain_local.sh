#!/usr/bin/env bash
# Snapshot queue: multi-table-trained PRPO, epochs 1--5 plus base.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
GPU="${1:?usage: $0 0|1}"
export FINQA_PRPO_RAW_ROOT="$DIR/checkpoints/finqa-multitable-v2-prpo-b64-24"
export FINQA_MULTITABLE_OUTPUT_ROOT="$DIR/outputs/prpo-multitrain-b64-24-local"
export FINQA_MULTITABLE_RUNTIME_TAG=prpo-multitrain-local
export FINQA_MULTITABLE_MERGED_CACHE=/mnt/disk1t/finqa-multitrain-prpo-eval-cache
export FINQA_MULTITABLE_BASE_MODEL=/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
case "$GPU" in
    0) PORT=29820; ITEMS=(base global_step_30 global_step_60) ;;
    1) PORT=29821; ITEMS=(global_step_15 global_step_45 global_step_75) ;;
    *) echo 'GPU must be 0 or 1' >&2; exit 2 ;;
esac
exec bash "$DIR/eval_lane.sh" prpo "$GPU" "$PORT" "${ITEMS[@]}"

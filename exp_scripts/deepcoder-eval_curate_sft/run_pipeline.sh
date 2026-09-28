#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="${1:?usage: run_pipeline.sh eval|curate|sft|all [RUN_NAME]}"
RUN_NAME="${2:-base-train-k8}"
WORK_ROOT="${WORK_ROOT:-/home/yanan/.deepcoder-sft-pipeline}"
EVAL_DIR="$WORK_ROOT/eval_runs/$RUN_NAME"
DATA_DIR="${DATA_DIR:-${EVAL_DIR}-sft-data}"
SFT_OUT="${SFT_OUT:-/home/yanan/.deepcoder-checkpoints/deepcoder-self-sft-qwen3-4b-r32}"

run_eval() {
  WORK_ROOT="$WORK_ROOT" RUN_NAME="$RUN_NAME" "$HERE/run_eval.sh"
}

run_curate() {
  "$HERE/run_curate.sh" "$EVAL_DIR" "$DATA_DIR"
}

run_sft() {
  "$HERE/run_sft.sh" "$DATA_DIR" "$SFT_OUT"
}

case "$STAGE" in
  eval) run_eval ;;
  curate) run_curate ;;
  sft) run_sft ;;
  all)
    run_eval
    run_curate
    run_sft
    ;;
  *)
    echo "unknown stage: $STAGE (expected eval|curate|sft|all)" >&2
    exit 2
    ;;
esac

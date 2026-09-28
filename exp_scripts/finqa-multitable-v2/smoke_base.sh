#!/usr/bin/env bash
# Two real multi-table tasks: one ordinary and one historically turn-cap-heavy.
set -euo pipefail

PROTOCOL_DIR="$(cd "$(dirname "$0")" && pwd)"
GRPO_DIR="$(cd "$PROTOCOL_DIR/../finqa-grpo-run" && pwd)"
RLLM_DIR=/home/yanan/agents/rllm
GPU="${1:-0}"
PORT="${2:-8610}"
TASK_INDICES="${SMOKE_TASK_INDICES:-0,115}"
EXPECTED="${SMOKE_EXPECTED:-2}"
OUT="$PROTOCOL_DIR/smoke/base_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

source "$GRPO_DIR/env.sh"
source "$VENV/bin/activate"
export PYTHONPATH="$PROTOCOL_DIR:$RLLM_DIR/cookbooks/finqa${PYTHONPATH:+:$PYTHONPATH}"
export FINQA_MULTI_TABLE_JUDGE_MODEL=gpt-5.4-nano

RUNTIME="$OUT/runtime"
SOURCE="$GRPO_DIR/.rllm_multilane0_multi_val/datasets"
mkdir -p "$RUNTIME/datasets/finqa"
cp "$SOURCE/registry.json" "$RUNTIME/datasets/registry.json"
cp "$SOURCE/finqa/"*.parquet "$RUNTIME/datasets/finqa/"

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$GPU" vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --port "$PORT" --max-model-len 49152 --max-num-seqs 2 \
    --gpu-memory-utilization 0.88 --tensor-parallel-size 1 \
    --generation-config vllm \
    --enable-auto-tool-choice --tool-call-parser hermes \
    >"$OUT/vllm_serve.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "vLLM died" >&2; exit 1; }
    sleep 5
done
curl -sf "http://localhost:$PORT/v1/models" >/dev/null

cd "$RLLM_DIR/cookbooks/finqa"
FINQA_JUDGE_FINISH_LOG="$OUT/judge_finish.tsv" RLLM_HOME="$RUNTIME" \
rllm eval finqa \
    --agent multitable_v2_flow:finqa_multitable_v2 --evaluator finqa \
    --model Qwen/Qwen3-4B-Instruct-2507 --base-url "http://localhost:$PORT/v1" \
    --split multi_val --task-indices "$TASK_INDICES" --concurrency 2 --attempts 1 \
    --sampling-params "temperature=0,top_p=1.0,seed=1234" \
    --episodes-dir "$OUT/episodes" --output "$OUT/result.json" \
    >"$OUT/eval.log" 2>&1
python "$PROTOCOL_DIR/audit_result.py" "$OUT/result.json" "$OUT/episodes" "$EXPECTED" | tee "$OUT/audit.log"
echo "SMOKE_COMPLETE out=$OUT"

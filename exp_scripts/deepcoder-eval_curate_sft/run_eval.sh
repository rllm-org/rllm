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

MODEL="${MODEL:-/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
GPU="${GPU:-0}"
PORT="${PORT:-8992}"
ATTEMPTS="${ATTEMPTS:-8}"
CONCURRENCY="${CONCURRENCY:-8}"
FLUSH_TASKS="${FLUSH_TASKS:-32}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
WORK_ROOT="${WORK_ROOT:-/home/yanan/.deepcoder-sft-pipeline}"
RUN_NAME="${RUN_NAME:-base-train-k${ATTEMPTS}}"
RUN_DIR="$WORK_ROOT/eval_runs/$RUN_NAME"

test -f "$MODEL/model.safetensors.index.json"
if [[ -z "$MAX_EXAMPLES" ]]; then
  echo "set MAX_EXAMPLES=N for a bounded run, or MAX_EXAMPLES=all explicitly" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to overwrite $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$WORK_ROOT/eval_runs"

export CUDA_VISIBLE_DEVICES="$GPU"
cd /home/yanan/agents/rllm
setsid vllm serve "$MODEL" --served-model-name deepcoder-sft-source \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 \
  --max-model-len 32768 --gpu-memory-utilization 0.9 --max-num-seqs 8 \
  >"$WORK_ROOT/${RUN_NAME}.server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill -9 -- "-$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for ((i=0; i<300; i++)); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "vLLM exited before ready" >&2; exit 1; }
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  sleep 2
done
[[ "$ready" == 1 ]]

args=(
  --url "http://127.0.0.1:$PORT/v1"
  --model deepcoder-sft-source
  --output "$RUN_DIR"
  --split train
  --attempts "$ATTEMPTS"
  --concurrency "$CONCURRENCY"
  --selection-seed 1234
  --num-shards "$NUM_SHARDS"
  --shard-index "$SHARD_INDEX"
  --temperature 0.6
  --top-p 1.0
  --max-tokens 16384
  --flush-tasks "$FLUSH_TASKS"
)
if [[ "$MAX_EXAMPLES" != all ]]; then args+=(--max-examples "$MAX_EXAMPLES"); fi
python -u "$HERE/eval_rollouts.py" "${args[@]}" | tee "$WORK_ROOT/${RUN_NAME}.eval.log"

echo "$RUN_DIR"

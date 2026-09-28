#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
GPU="${1:?GPU}"; PORT="${2:?port}"; OUT="${3:?output root}"
source "$ROOT/../finqa-grpo-run/env.sh"
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPU" RLLM_HOME="$ROOT/runtime"
export PYTHONPATH="$ROOT:/home/yanan/agents/rllm/cookbooks/deepcoder:/home/yanan/agents/rllm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Defaults keep the historical behaviour (Qwen3-4B-Instruct-2507, 2 repeats).
# MODEL/REPEATS/CHAT_TEMPLATE let the same lane measure another base model under
# the identical protocol; everything else (sampling, task set, grader) lives in
# eval_base.py and is asserted there.
MODEL="${MODEL:-/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
REPEATS="${REPEATS:-2}"
EXTRA_SERVE_ARGS=()
if [[ -n "${CHAT_TEMPLATE:-}" ]]; then
  test -f "$CHAT_TEMPLATE"
  EXTRA_SERVE_ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi
test -f "$MODEL/model.safetensors.index.json"
mkdir -p "$OUT"
cd /home/yanan/agents/rllm
setsid vllm serve "$MODEL" --served-model-name deepcoder-base --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.9 --max-num-seqs "${EVAL_CONCURRENCY:-8}" "${EXTRA_SERVE_ARGS[@]}" > "$OUT/gpu${GPU}-server.log" 2>&1 &
SERVER_PID=$!
# vLLM v1 re-parents EngineCore after the API process dies; kill the whole process group
# (setsid above) or ~29 GiB stays pinned on the GPU and the next server fails to start.
trap 'kill -9 -- "-$SERVER_PID" 2>/dev/null || kill -9 "$SERVER_PID" 2>/dev/null || true' EXIT
READY=0
for ((i=0;i<300;i++)); do
  kill -0 "$SERVER_PID" || exit 1
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then READY=1; break; fi
  sleep 2
done
[[ "$READY" == 1 ]]
for ((REP=1; REP<=REPEATS; REP++)); do
  python -u "$ROOT/eval_base.py" --url "http://127.0.0.1:$PORT/v1" --model deepcoder-base --output "$OUT/gpu${GPU}-repeat${REP}" > "$OUT/gpu${GPU}-repeat${REP}.log" 2>&1
done

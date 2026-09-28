#!/usr/bin/env bash
# One base-model measurement on the 687-task DeepCoder test split, split across
# both local GPUs (343 + 344 tasks) and merged back. Roughly halves wall-clock
# versus one GPU doing all 687.
#
#   MODEL=<hf_dir> [CHAT_TEMPLATE=<jinja>] ./eval_base_dual_sharded.sh <out_dir>
#
# Protocol lives in eval_base.py and is asserted there: split=test, temperature=0,
# top_p=1.0, seed=1234, max_tokens=16384, max_model_len=32768.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:?usage: MODEL=... $0 <out_dir>}"
MODEL="${MODEL:?set MODEL to a local HF model dir}"
PORT0="${PORT0:-28120}"; PORT1="${PORT1:-28121}"
export EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-32}"
# flashinfer's sampling JIT does not build against this CUDA/cub combination;
# with temperature=0 the sampler is argmax either way.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

source "$ROOT/../finqa-grpo-run/env.sh"
source "$VENV/bin/activate"
export RLLM_HOME="$ROOT/runtime"
export PYTHONPATH="$ROOT:/home/yanan/agents/rllm/cookbooks/deepcoder:/home/yanan/agents/rllm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
test -f "$MODEL/model.safetensors.index.json"
mkdir -p "$OUT"

SERVE_EXTRA=()
if [[ -n "${CHAT_TEMPLATE:-}" ]]; then test -f "$CHAT_TEMPLATE"; SERVE_EXTRA+=(--chat-template "$CHAT_TEMPLATE"); fi

PGIDS=()
cleanup() {
  # vLLM v1 re-parents EngineCore; kill the group or ~29 GiB stays pinned.
  for pg in "${PGIDS[@]:-}"; do [[ -n "$pg" ]] && kill -9 -- "-$pg" 2>/dev/null || true; done
}
trap cleanup EXIT

start_server() {  # $1=gpu $2=port
  cd /home/yanan/agents/rllm
  CUDA_VISIBLE_DEVICES="$1" setsid vllm serve "$MODEL" --served-model-name deepcoder-base \
    --host 127.0.0.1 --port "$2" --tensor-parallel-size 1 --max-model-len 32768 \
    --gpu-memory-utilization 0.9 --max-num-seqs "$EVAL_CONCURRENCY" "${SERVE_EXTRA[@]}" \
    > "$OUT/gpu$1-server.log" 2>&1 &
  echo $!
}
P0=$(start_server 0 "$PORT0"); PGIDS+=("$P0")
P1=$(start_server 1 "$PORT1"); PGIDS+=("$P1")
echo "[dual] servers pid0=$P0 pid1=$P1"

for spec in "$PORT0:$P0" "$PORT1:$P1"; do
  port="${spec%%:*}"; pid="${spec##*:}"; ready=0
  for ((i=0; i<360; i++)); do
    kill -0 "$pid" 2>/dev/null || { echo "[dual] vLLM $pid died; see $OUT/gpu*-server.log" >&2; exit 1; }
    curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    sleep 5
  done
  [[ "$ready" == 1 ]] || { echo "[dual] port $port never ready" >&2; exit 1; }
done
echo "[dual] both servers ready"

run_shard() {  # $1=shard index $2=port
  python -u "$ROOT/eval_base.py" --url "http://127.0.0.1:$2/v1" --model deepcoder-base \
    --output "$OUT/shard$1" --num-shards 2 --shard-index "$1" > "$OUT/shard$1.log" 2>&1
}
run_shard 0 "$PORT0" & S0=$!
run_shard 1 "$PORT1" & S1=$!
rc=0; wait "$S0" || rc=1; wait "$S1" || rc=1
if [[ "$rc" != 0 ]]; then echo "[dual] a shard failed; see $OUT/shard*.log" >&2; exit 1; fi

python -u "$ROOT/merge_shards.py" --shard "$OUT/shard0" --shard "$OUT/shard1" --output "$OUT" | tee "$OUT/merge.log"
echo "[dual] DONE $(date '+%F %T %Z')"

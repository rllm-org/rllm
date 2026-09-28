#!/usr/bin/env bash
# Evaluate a queue of FinQA PRPO models on one local GPU.  Each model is served
# once; val (522) and test (558) are evaluated sequentially against it.
# Scientific protocol matches finqa-grpo-run/eval_checkpoint_lane.sh.
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
GRPO_DIR="$(cd "$RUN_DIR/../finqa-grpo-run" && pwd)"
GPU="${1:?usage: $0 GPU PORT base|global_step_N [...] }"
PORT="${2:?usage: $0 GPU PORT base|global_step_N [...] }"
shift 2
[[ "$#" -gt 0 ]] || { echo "no models supplied" >&2; exit 2; }

source "$GRPO_DIR/env.sh"
source "$VENV/bin/activate"
cd /home/yanan/agents/rllm/cookbooks/finqa

RAW_ROOT="${PRPO_RAW_ROOT:-$RUN_DIR/checkpoints/qwen3-4b-16-prpo-sync-b64-formal}"
MERGED_ROOT="${PRPO_MERGED_ROOT:-$RUN_DIR/merged}"
EVAL_ROOT="${PRPO_EVAL_ROOT:-$RUN_DIR/eval_greedy}"
RUNTIME_ROOT="${PRPO_RUNTIME_ROOT:-$RUN_DIR/runtime_eval/lane${GPU}}"
mkdir -p "$MERGED_ROOT" "$EVAL_ROOT" "$RUNTIME_ROOT"

complete_output() {
  python - "$1" "$2" <<'PY'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
try:
    data = json.load(open(path))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if data.get("total") == expected and len(data.get("items", [])) == expected else 1)
PY
}

prepare_home() {
  local split="$1"
  local home="$RUNTIME_ROOT/$split"
  # Use a lane-local registry containing parquet only.  In particular, never
  # copy dataset.toml/data: that materialised path makes rLLM ignore --split.
  find "$home" -depth -mindepth 1 -delete 2>/dev/null || true
  mkdir -p "$home/datasets/finqa"
  cp "$HOME/.rllm/datasets/registry.json" "$home/datasets/" 2>/dev/null || true
  cp "$HOME"/.rllm/datasets/finqa/*.parquet "$home/datasets/finqa/"
}

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

for ITEM in "$@"; do
  if [[ "$ITEM" == base ]]; then
    MODEL="${PRPO_BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
    TAG="prpo_base"
  else
    RAW="$RAW_ROOT/$ITEM"
    MODEL="$MERGED_ROOT/$ITEM"
    TAG="prpo_${ITEM}"
    [[ -d "$RAW/actor" ]] || { echo "[$TAG] incomplete raw checkpoint: $RAW" >&2; exit 1; }
    if [[ ! -f "$MODEL/config.json" ]] || ! find "$MODEL" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q .; then
      mkdir -p "$EVAL_ROOT/$TAG"
      echo "[$TAG] merging $RAW -> $MODEL"
      python "$GRPO_DIR/merge_lora.py" "$RAW" "$MODEL" >"$EVAL_ROOT/$TAG/merge.log" 2>&1
    fi
  fi

  OUT="$EVAL_ROOT/$TAG"
  mkdir -p "$OUT"
  echo "[$TAG] serving model=$MODEL gpu=$GPU port=$PORT"
  CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" \
    --port "$PORT" --max-model-len 12288 \
    --gpu-memory-utilization 0.85 --tensor-parallel-size 1 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    >"$OUT/vllm_serve.log" 2>&1 &
  SERVER_PID=$!

  ready=0
  for _ in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "[$TAG] vLLM died" >&2; exit 1; }
    sleep 5
  done
  [[ "$ready" -eq 1 ]] || { echo "[$TAG] vLLM readiness timeout" >&2; exit 1; }

  for SPLIT in val test; do
    if [[ "$SPLIT" == val ]]; then EXPECTED=522; else EXPECTED=558; fi
    if complete_output "$OUT/$SPLIT.json" "$EXPECTED"; then
      echo "[$TAG] $SPLIT already complete; skipping"
      continue
    fi
    prepare_home "$SPLIT"
    echo "[$TAG] evaluating $SPLIT ($EXPECTED rows), greedy"
    FINQA_JUDGE_FINISH_LOG="$OUT/judge_finish_${SPLIT}.tsv" \
    RLLM_HOME="$RUNTIME_ROOT/$SPLIT" \
    rllm eval finqa \
      --agent finqa --evaluator finqa \
      --model "$MODEL" --base-url "http://localhost:$PORT/v1" \
      --split "$SPLIT" --concurrency 32 \
      --sampling-params "temperature=0,top_p=1.0,seed=1234" \
      --episodes-dir "$OUT/episodes_$SPLIT" \
      --output "$OUT/$SPLIT.json" >"$OUT/eval_$SPLIT.log" 2>&1
    complete_output "$OUT/$SPLIT.json" "$EXPECTED" || { echo "[$TAG] incomplete $SPLIT output" >&2; exit 1; }
    python - "$OUT/$SPLIT.json" "$TAG" "$SPLIT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[{sys.argv[2]}] {sys.argv[3]}: {d['correct']}/{d['total']} score={d['score']:.6f} errors={d['errors']}")
PY
  done
  stop_server
  echo "[$TAG] COMPLETE"
done

echo "LANE_COMPLETE gpu=$GPU"

#!/usr/bin/env bash
# Runtime queue for evaluating raw verl checkpoints on one GPU.
# Evaluation semantics intentionally match eval_parallel.sh; only scheduling
# differs: val and test run sequentially against one resident vLLM server.
set -euo pipefail

# Tool-call/reasoning parser flags are derived from the model's chat template;
# a hardcoded parser silently drops every tool call on a model that uses another format.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_parser_flags.sh"

RD="$(cd "$(dirname "$0")" && pwd)"
GPU="${1:?usage: $0 GPU PORT global_step_N [...]}"
PORT="${2:?usage: $0 GPU PORT global_step_N [...]}"
shift 2
[ "$#" -gt 0 ] || { echo "no checkpoints supplied" >&2; exit 2; }

EVAL_SEED=1234
EVAL_TEMP=0
EVAL_TOP_P=1.0
# Optional switches (unset = the original protocol):
#   LORA_ADAPTER_DIR=<dir>  serve the base model with <dir>/<step>_verl_raw/lora_adapter mounted
#                           (--enable-lora) instead of the bf16-merged model; results go to eval/lora_<step>.
#                           Extract the adapter first with `python -m verl.model_merger merge --backend fsdp`.
#   GPU_UTIL=<fraction>     --gpu-memory-utilization (default 0.85).
LORA_ADAPTER_DIR="${LORA_ADAPTER_DIR:-}"
GPU_UTIL="${GPU_UTIL:-0.85}"
BASE_SNAPSHOT=/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554

source "$RD/env.sh"
source "$VENV/bin/activate"
cd /home/yanan/agents/rllm/cookbooks/finqa

checkpoint_complete() {
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
    local home="$RD/.rllm_lane${GPU}_${split}"
    rm -rf "$home"
    mkdir -p "$home/datasets/finqa"
    cp "$HOME/.rllm/datasets/registry.json" "$home/datasets/" 2>/dev/null || true
    cp "$HOME"/.rllm/datasets/finqa/*.parquet "$home/datasets/finqa/"
}

SERVER_PID=""
stop_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT INT TERM

for STEP in "$@"; do
    RAW="$RD/checkpoints/qwen3-4b-16-epoch/$STEP"
    MODEL="$RD/merged/$STEP"
    TAG="greedy_$STEP"
    SERVED="$MODEL"
    LORA_ARGS=()
    if [ -n "$LORA_ADAPTER_DIR" ]; then
        ADAPTER="$LORA_ADAPTER_DIR/${STEP}_verl_raw/lora_adapter"
        [ -f "$ADAPTER/adapter_config.json" ] || { echo "[$STEP] missing adapter: $ADAPTER" >&2; exit 1; }
        MODEL="$BASE_SNAPSHOT"
        TAG="lora_$STEP"
        SERVED=ckpt
        LORA_ARGS=(--enable-lora --max-lora-rank 32 --max-loras 1 --lora-modules "ckpt=$ADAPTER")
    fi
    OUT="$RD/eval/$TAG"
    [ -d "$RAW/actor" ] || { echo "[$TAG] missing raw checkpoint: $RAW" >&2; exit 1; }
    mkdir -p "$OUT"

    if [ -n "$LORA_ADAPTER_DIR" ]; then
        echo "[$TAG] base $MODEL + adapter $ADAPTER (no merge)"
    elif [ ! -f "$MODEL/config.json" ] || ! find "$MODEL" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q .; then
        echo "[$TAG] merging $RAW -> $MODEL"
        python "$RD/merge_lora.py" "$RAW" "$MODEL" > "$OUT/merge.log" 2>&1
    else
        echo "[$TAG] reusing complete merged model $MODEL"
    fi

    echo "[$TAG] starting vLLM on GPU=$GPU port=$PORT"
    resolve_parser_flags "$MODEL"
    CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" \
        --port "$PORT" --max-model-len 12288 \
        --gpu-memory-utilization "$GPU_UTIL" --tensor-parallel-size 1 \
        "${PARSER_FLAGS[@]}" "${LORA_ARGS[@]}" \
        > "$OUT/vllm_serve.log" 2>&1 &
    SERVER_PID=$!

    ready=0
    for _ in $(seq 1 180); do
        if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "[$TAG] vLLM died" >&2; exit 1; }
        sleep 5
    done
    [ "$ready" -eq 1 ] || { echo "[$TAG] vLLM readiness timeout" >&2; exit 1; }

    for SPLIT in val test; do
        if [ "$SPLIT" = val ]; then EXPECTED=522; else EXPECTED=558; fi
        if checkpoint_complete "$OUT/$SPLIT.json" "$EXPECTED"; then
            echo "[$TAG] $SPLIT already complete; skipping"
            continue
        fi
        prepare_home "$SPLIT"
        echo "[$TAG] evaluating $SPLIT ($EXPECTED rows)"
        FINQA_JUDGE_FINISH_LOG="$OUT/judge_finish_${SPLIT}.tsv" \
        RLLM_HOME="$RD/.rllm_lane${GPU}_${SPLIT}" \
        rllm eval finqa \
            --agent finqa --evaluator finqa \
            --model "$SERVED" --base-url "http://localhost:$PORT/v1" \
            --split "$SPLIT" --concurrency 32 \
            --sampling-params "temperature=$EVAL_TEMP,top_p=$EVAL_TOP_P,seed=$EVAL_SEED" \
            --episodes-dir "$OUT/episodes_$SPLIT" \
            --output "$OUT/$SPLIT.json" > "$OUT/eval_$SPLIT.log" 2>&1
        checkpoint_complete "$OUT/$SPLIT.json" "$EXPECTED" || {
            echo "[$TAG] $SPLIT output failed completeness check" >&2
            exit 1
        }
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

#!/usr/bin/env bash
# Evaluate FinQA multi-table val/test for a queue of models on one GPU.
#
# Policy decoding matches the established offline single-table protocol:
#   temperature=0, top_p=1.0, seed=1234, one rollout per task.
# Multi-table scoring is selected by question_type in finqa_eval.py. This
# queue explicitly uses gpt-5.4-nano and gives every model/split an independent
# output directory and judge finish-reason log.
set -euo pipefail

# Tool-call/reasoning parser flags are derived from the model's chat template;
# a hardcoded parser silently drops every tool call on a model that uses another format.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_parser_flags.sh"

RD="$(cd "$(dirname "$0")" && pwd)"
GPU="${1:?missing GPU argument}"
PORT="${2:?missing PORT argument}"
shift 2
[ "$#" -gt 0 ] || { echo "no models supplied" >&2; exit 2; }

EVAL_SEED=1234
EVAL_TEMP=0
EVAL_TOP_P=1.0

source "$RD/env.sh"
source "$VENV/bin/activate"
cd /home/yanan/agents/rllm/cookbooks/finqa

# This experiment deliberately uses the same judge tier as single-table eval.
# Keep the override local to these lane processes; do not change env.sh's
# default for unrelated multi-table runs.
export FINQA_MULTI_TABLE_JUDGE_MODEL=gpt-5.4-nano

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
    local home="$RD/.rllm_multilane${GPU}_${split}"
    rm -rf "$home"
    mkdir -p "$home/datasets/finqa"
    cp "$HOME/.rllm/datasets/registry.json" "$home/datasets/" 2>/dev/null || true
    cp "$HOME"/.rllm/datasets/finqa/*.parquet "$home/datasets/finqa/"
    [ -f "$home/datasets/finqa/${split}.parquet" ] || {
        echo "missing registered split parquet: $split" >&2
        exit 1
    }
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

for ITEM in "$@"; do
    if [ "$ITEM" = base ]; then
        MODEL=Qwen/Qwen3-4B-Instruct-2507
        TAG=greedy_multi_base_d16
    else
        RAW="$RD/checkpoints/qwen3-4b-16-epoch/$ITEM"
        MODEL="$RD/merged/$ITEM"
        TAG="greedy_multi_$ITEM"
        [ -d "$RAW/actor" ] || { echo "[$TAG] missing raw checkpoint: $RAW" >&2; exit 1; }
        [ -f "$MODEL/config.json" ] || { echo "[$TAG] missing merged config: $MODEL/config.json" >&2; exit 1; }
        find "$MODEL" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q . || {
            echo "[$TAG] missing merged model weights: $MODEL" >&2
            exit 1
        }
    fi

    OUT="$RD/eval/$TAG"
    mkdir -p "$OUT"
    echo "[$TAG] starting vLLM on GPU=$GPU port=$PORT model=$MODEL"
    resolve_parser_flags "$MODEL"
    CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" \
        --port "$PORT" --max-model-len 12288 \
        --gpu-memory-utilization 0.85 --tensor-parallel-size 1 \
        "${PARSER_FLAGS[@]}" \
        > "$OUT/vllm_serve.log" 2>&1 &
    SERVER_PID=$!

    ready=0
    for _ in $(seq 1 180); do
        if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "[$TAG] vLLM died; see $OUT/vllm_serve.log" >&2; exit 1; }
        sleep 5
    done
    [ "$ready" -eq 1 ] || { echo "[$TAG] vLLM readiness timeout" >&2; exit 1; }

    for SPLIT in multi_val multi_test; do
        if [ "$SPLIT" = multi_val ]; then EXPECTED=126; else EXPECTED=131; fi
        if checkpoint_complete "$OUT/$SPLIT.json" "$EXPECTED"; then
            echo "[$TAG] $SPLIT already complete; skipping"
            continue
        fi
        prepare_home "$SPLIT"
        echo "[$TAG] evaluating $SPLIT ($EXPECTED rows)"
        FINQA_JUDGE_FINISH_LOG="$OUT/judge_finish_${SPLIT}.tsv" \
        RLLM_HOME="$RD/.rllm_multilane${GPU}_${SPLIT}" \
        rllm eval finqa \
            --agent finqa --evaluator finqa \
            --model "$MODEL" --base-url "http://localhost:$PORT/v1" \
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

echo "MULTITABLE_LANE_COMPLETE gpu=$GPU"

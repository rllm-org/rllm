#!/usr/bin/env bash
# Evaluate one model on the single-table FinQA eval sets.
#
#   ./eval_full.sh base
#   SPLITS="val test" ./eval_full.sh <ckpt> tag
#   ./eval_full.sh /path/to/merged_hf_checkpoint  [tag]
#
# Splits:
#   val        522  single-table  (same set the training loop validates on)
#   test       558  single-table
# The 12288-token context here is sized for single-table episodes only. Multi-table
# evaluation uses finqa-multitable-v2/eval_lane.sh (49152-token context).
set -euo pipefail

# Tool-call/reasoning parser flags are derived from the model's chat template;
# a hardcoded parser silently drops every tool call on a model that uses another format.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_parser_flags.sh"
RD="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:?usage: $0 {base|/path/to/merged_hf} [tag]}"
PORT="${PORT:-8100}"      # stay well below the 32768 ephemeral floor

source "$RD/env.sh"
source "$VENV/bin/activate"
cd /home/yanan/agents/rllm/cookbooks/finqa

if [ "$TARGET" = "base" ]; then
    MODEL=Qwen/Qwen3-4B-Instruct-2507
    TAG="${2:-base}"
elif [ -d "$TARGET/actor" ]; then
    # A raw verl checkpoint. It CANNOT be served directly: verl's model_merger
    # strips the LoRA keys out of the state dict and writes them to a separate
    # lora_adapter/ dir, so target_dir/ holds BASE weights only. Serving that
    # would silently evaluate the base model and report it as the trained score.
    # merge_lora.py applies the adapter and refuses to emit anything unless the
    # weights provably changed.
    TAG="${2:-trained}"
    MODEL="$RD/merged/$(basename "$TARGET")"
    if [ ! -d "$MODEL" ]; then
        echo "[eval_full] merging LoRA checkpoint -> $MODEL"
        python "$RD/merge_lora.py" "$TARGET" "$MODEL" || {
            echo "[eval_full] merge failed; refusing to run an eval that would score the base model" >&2
            exit 1
        }
    else
        echo "[eval_full] reusing already-merged $MODEL"
    fi
else
    MODEL="$TARGET"
    TAG="${2:-trained}"
fi
OUT="$RD/eval/$TAG"; mkdir -p "$OUT"

echo "[eval_full] model=$MODEL tag=$TAG"
# finqa_flow hardcodes api_key="EMPTY", so the server must not require auth.
resolve_parser_flags "$MODEL"
nohup vllm serve "$MODEL" \
    --port "$PORT" --max-model-len 12288 \
    --gpu-memory-utilization 0.85 --tensor-parallel-size 1 \
    "${PARSER_FLAGS[@]}" \
    > "$OUT/vllm_serve.log" 2>&1 &
VLLM_PID=$!
trap 'echo "[eval_full] stopping vllm $VLLM_PID"; kill "$VLLM_PID" 2>/dev/null || true' EXIT
echo "[eval_full] vllm pid=$VLLM_PID, waiting for readiness ..."
for _ in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "[eval_full] vllm died; see $OUT/vllm_serve.log" >&2; exit 1; }
    sleep 5
done
curl -sf "http://localhost:$PORT/v1/models" >/dev/null || { echo "[eval_full] vllm never ready" >&2; exit 1; }
echo "[eval_full] vLLM ready"

for SPLIT in ${SPLITS:-val test}; do
    echo "=== [$TAG] split=$SPLIT ==="
    rllm eval finqa \
        --agent finqa --evaluator finqa \
        --model "$MODEL" --base-url "http://localhost:$PORT/v1" \
        --split "$SPLIT" \
        --concurrency 32 \
        --episodes-dir "$OUT/episodes_$SPLIT" \
        --output "$OUT/${SPLIT}.json" 2>&1 | tail -20
done
echo "EVAL_FULL_DONE tag=$TAG"

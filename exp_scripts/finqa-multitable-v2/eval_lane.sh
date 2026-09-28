#!/usr/bin/env bash
# Evaluate one arm/model queue under the isolated FinQA multi-table-v2 protocol.
set -euo pipefail

PROTOCOL_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "$PROTOCOL_DIR/.." && pwd)"
GRPO_DIR="$WORKSPACE_DIR/finqa-grpo-run"
PRPO_DIR="$WORKSPACE_DIR/finqa-prpo-run"
RLLM_DIR=/home/yanan/agents/rllm

ARM="${1:?usage: $0 grpo|prpo|rpp GPU PORT base|global_step_N [...] }"
GPU="${2:?missing GPU}"
PORT="${3:?missing port}"
shift 3
[[ "$ARM" == grpo || "$ARM" == prpo || "$ARM" == rpp ]] || { echo "invalid arm: $ARM" >&2; exit 2; }
[[ "$#" -gt 0 ]] || { echo "no models supplied" >&2; exit 2; }

source "$GRPO_DIR/env.sh"
source "$VENV/bin/activate"
export PYTHONPATH="$PROTOCOL_DIR:$RLLM_DIR/cookbooks/finqa${PYTHONPATH:+:$PYTHONPATH}"
cd "$RLLM_DIR/cookbooks/finqa"

export FINQA_MULTI_TABLE_JUDGE_MODEL=gpt-5.4-nano

OUTPUT_ROOT="${FINQA_MULTITABLE_OUTPUT_ROOT:-$PROTOCOL_DIR/outputs/$ARM}"
RUNTIME_TAG="${FINQA_MULTITABLE_RUNTIME_TAG:-local}"
[[ "$RUNTIME_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid runtime tag: $RUNTIME_TAG" >&2; exit 2; }
RUNTIME_ROOT="$PROTOCOL_DIR/runtime/${RUNTIME_TAG}/lane${GPU}"
MERGED_CACHE="${FINQA_MULTITABLE_MERGED_CACHE:-/mnt/disk1t/finqa-multitable-v2-merged-cache}"
DATA_SOURCE="$GRPO_DIR/.rllm_multilane0_multi_val/datasets"
mkdir -p "$OUTPUT_ROOT" "$RUNTIME_ROOT" "$MERGED_CACHE"

prepare_home() {
    local split="$1"
    local runtime_home="$RUNTIME_ROOT/$split"
    mkdir -p "$runtime_home/datasets/finqa"
    cp "$DATA_SOURCE/registry.json" "$runtime_home/datasets/registry.json"
    cp "$DATA_SOURCE/finqa/"*.parquet "$runtime_home/datasets/finqa/"
    [[ -f "$runtime_home/datasets/finqa/${split}.parquet" ]] || {
        echo "missing registered split: $split" >&2
        exit 1
    }
}

resolve_model() {
    local item="$1"
    MERGED_WAS_CREATED=0
    if [[ "$item" == base ]]; then
        MODEL="${FINQA_MULTITABLE_BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
        TAG=base
        return
    fi
    [[ "$item" =~ ^global_step_[0-9]+$ ]] || { echo "invalid checkpoint name: $item" >&2; exit 2; }
    TAG="$item"
    if [[ "$ARM" == grpo ]]; then
        MODEL="$GRPO_DIR/merged/$item"
        [[ -f "$MODEL/config.json" ]] || { echo "missing GRPO merged model: $MODEL" >&2; exit 1; }
    elif [[ "$ARM" == prpo ]]; then
        local raw="${FINQA_PRPO_RAW_ROOT:-$PRPO_DIR/checkpoints/qwen3-4b-16-prpo-sync-b64-formal}/$item"
        if [[ ! -d "$raw/actor" ]]; then
            [[ -z "${FINQA_PRPO_RAW_ROOT:-}" ]] || { echo "missing requested PRPO checkpoint: $raw" >&2; exit 1; }
            raw="/mnt/disk1t/finqa-prpo-run-checkpoints/raw/$item"
        fi
        [[ -d "$raw/actor" ]] || { echo "missing PRPO raw checkpoint: $item" >&2; exit 1; }
        MODEL="$MERGED_CACHE/prpo_$item"
        if [[ ! -f "$MODEL/config.json" ]] || ! find "$MODEL" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q .; then
            mkdir -p "$OUTPUT_ROOT/$TAG"
            python "$GRPO_DIR/merge_lora.py" "$raw" "$MODEL" >"$OUTPUT_ROOT/$TAG/merge.log" 2>&1
            MERGED_WAS_CREATED=1
        fi
    else
        local raw="${FINQA_RPP_RAW_ROOT:?FINQA_RPP_RAW_ROOT is required for arm=rpp}/$item"
        [[ -d "$raw/actor" ]] || { echo "missing RPP raw checkpoint: $item" >&2; exit 1; }
        MODEL="$MERGED_CACHE/rpp_$item"
        if [[ ! -f "$MODEL/config.json" ]] || ! find "$MODEL" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q .; then
            mkdir -p "$OUTPUT_ROOT/$TAG"
            python "$GRPO_DIR/merge_lora.py" "$raw" "$MODEL" >"$OUTPUT_ROOT/$TAG/merge.log" 2>&1
            MERGED_WAS_CREATED=1
        fi
    fi
    find "$MODEL" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q . || {
        echo "model has no safetensors weights: $MODEL" >&2
        exit 1
    }
}

write_manifest() {
    local out="$1" item="$2"
    python - "$out/protocol_manifest.json" "$ARM" "$item" "$MODEL" "$GPU" "$PORT" <<'PY'
import hashlib, json, pathlib, sys
out, arm, item, model, gpu, port = sys.argv[1:]
root = pathlib.Path("/home/yanan/agents/rllm/exp_scripts/finqa-multitable-v2")
judge = pathlib.Path("/home/yanan/agents/rllm/cookbooks/finqa/prompts/multi_table_correctness_prompt.txt")
def digest(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
manifest = {
    "protocol_version": "finqa-multitable-visible-complete-v2",
    "arm": arm, "checkpoint": item, "model": model, "gpu": int(gpu), "port": int(port),
    "max_turns": 50, "max_tool_turns": 45, "reserved_final_attempts": 5,
    "serving_context_tokens": 49152, "discovery_max_completion_tokens": 2048,
    "final_max_completion_tokens": 8192, "tool_output_chars": 8000,
    "sampling": {"temperature": 0, "top_p": 1.0, "seed": 1234, "attempts": 1,
                 "vllm_generation_config": "vllm defaults (HF generation_config disabled)"},
    "judge_model": "gpt-5.4-nano", "judge_protocol": "upstream FinQA multi-table rubric unchanged",
    "flow_sha256": digest(root / "multitable_v2_flow.py"),
    "policy_prompt_sha256": digest(root / "prompts/multitable_v2_system_prompt.txt"),
    "judge_prompt_sha256": digest(judge),
}
pathlib.Path(out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
}

SERVER_PID=""
stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        if [[ "${FINQA_FORCE_KILL9:-0}" == 1 ]]; then
            kill -9 "$SERVER_PID" 2>/dev/null || true
        else
            kill "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT INT TERM

for ITEM in "$@"; do
    resolve_model "$ITEM"
    OUT="$OUTPUT_ROOT/$TAG"
    mkdir -p "$OUT"
    [[ ! -e "$OUT/COMPLETE" ]] || { echo "[$ARM/$TAG] already complete"; continue; }
    write_manifest "$OUT" "$ITEM"

    echo "[$ARM/$TAG] serving $MODEL on gpu=$GPU port=$PORT"
    CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" \
        --port "$PORT" --max-model-len 49152 --max-num-seqs 8 \
        --gpu-memory-utilization 0.88 --tensor-parallel-size 1 \
        --generation-config vllm \
        --enable-auto-tool-choice --tool-call-parser hermes \
        >"$OUT/vllm_serve.log" 2>&1 &
    SERVER_PID=$!
    ready=0
    for _ in $(seq 1 180); do
        if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "vLLM died: $OUT/vllm_serve.log" >&2; exit 1; }
        sleep 5
    done
    [[ "$ready" -eq 1 ]] || { echo "vLLM readiness timeout" >&2; exit 1; }

    for SPLIT in multi_val multi_test; do
        [[ "$SPLIT" == multi_val ]] && EXPECTED=126 || EXPECTED=131
        RESULT="$OUT/$SPLIT.json"
        EPISODES="$OUT/episodes_$SPLIT"
        if [[ -e "$RESULT" || -e "$EPISODES" ]]; then
            echo "[$ARM/$TAG] refusing to mix stale/incomplete v2 output for $SPLIT" >&2
            exit 1
        fi
        prepare_home "$SPLIT"
        echo "[$ARM/$TAG] evaluating $SPLIT ($EXPECTED tasks), greedy"
        FINQA_JUDGE_FINISH_LOG="$OUT/judge_finish_${SPLIT}.tsv" \
        RLLM_HOME="$RUNTIME_ROOT/$SPLIT" \
        rllm eval finqa \
            --agent multitable_v2_flow:finqa_multitable_v2 --evaluator finqa \
            --model "$MODEL" --base-url "http://localhost:$PORT/v1" \
            --split "$SPLIT" --concurrency 8 --attempts 1 \
            --sampling-params "temperature=0,top_p=1.0,seed=1234" \
            --episodes-dir "$EPISODES" --output "$RESULT" \
            >"$OUT/eval_$SPLIT.log" 2>&1
        python "$PROTOCOL_DIR/audit_result.py" "$RESULT" "$EPISODES" "$EXPECTED" \
            >"$OUT/audit_$SPLIT.log" 2>&1
    done
    stop_server
    touch "$OUT/COMPLETE"
    if [[ ( "$ARM" == prpo || "$ARM" == rpp ) && "$ITEM" != base ]]; then
        python - "$MODEL" <<'PY'
import shutil, sys
shutil.rmtree(sys.argv[1])
PY
    fi
    echo "[$ARM/$TAG] COMPLETE"
done

echo "V2_LANE_COMPLETE arm=$ARM gpu=$GPU"

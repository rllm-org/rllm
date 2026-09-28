#!/usr/bin/env bash
# Evaluate ONE model on val and test at the same time, one split per GPU.
#
#   ./eval_parallel.sh base
#   ./eval_parallel.sh <merged_hf_dir|checkpoint_global_step_dir> [tag]
#
# GPU 0 -> val (522)   GPU 1 -> test (558)
#
# Both splits run against the same model, the same judge (gpt-5.4-nano) and the
# same NFS venv the training uses, so numbers are comparable across models.
#
# Sampling is pinned to temperature=0.6 / top_p=0.95 / seed, matching the
# in-training validation (train_16.sh val_kwargs) and making runs repeatable.
# Unpinned, the eval inherits the server defaults (temperature 1.0) and two runs
# of the SAME base model on the SAME test rows disagreed on 14.9% of tasks.
#
# The seed matters more than usual here: step_62 differs from base by ~7e-7 in
# relative weight terms (only ~0.02% of elements moved by one bf16 ULP), so an
# unseeded 14.9% noise floor would bury any real effect. A fixed seed makes the
# comparison paired rather than independent. There is no --seed flag, but
# --sampling-params passes unknown keys straight through to the backend
# (rllm/cli/_sampling.py: SamplingConfig.extra).
set -euo pipefail

# Tool-call/reasoning parser flags are derived from the model's chat template;
# a hardcoded parser silently drops every tool call on a model that uses another format.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_parser_flags.sh"
RD="$(cd "$(dirname "$0")" && pwd)"
if [ $# -lt 1 ]; then
    echo "usage: $0 <base | /path/to/model | /path/to/global_step_N> [tag]" >&2
    exit 2
fi
TARGET="$1"
PORT_VAL="${PORT_VAL:-8110}"     # keep well below the 32768 ephemeral floor
PORT_TEST="${PORT_TEST:-8111}"
EVAL_SEED="${EVAL_SEED:-1234}"
# Greedy (temperature 0) is the DEFAULT here, and deliberately differs from the
# in-training validation protocol (0.6 / 0.95, which mirrors finqa.md).
#
# The two serve different purposes. In-training validation estimates the
# sampling policy's expected performance. Standalone checkpoint eval asks a
# different question -- are these two models different? -- and there the
# sampling variance is pure noise, not the quantity of interest.
#
# Measured on this box with the training protocol: three runs of the SAME base
# model over the SAME 522 val rows scored 341 / 337 / 319 correct, a 4.21 pt
# spread, 5-20x larger than any model-vs-base delta observed. A fixed seed does
# NOT fix this: continuous batching varies the reduction order, perturbing
# logits and forking trajectories. Greedy removes the dominant term; argmax is
# far more robust to those perturbations than sampling is.
#
# Consequence: greedy numbers are not comparable to the in-training val curve,
# nor to finqa.md's sampled Pass@1. They are only for model-vs-model comparison.
EVAL_TEMP="${EVAL_TEMP:-0}"      # greedy by default for checkpoint comparison
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"

source "$RD/env.sh"
source "$VENV/bin/activate"
cd /home/yanan/agents/rllm/cookbooks/finqa

if [ "$TARGET" = "base" ]; then
    MODEL=Qwen/Qwen3-4B-Instruct-2507
    TAG="${2:-base}"
elif [ -d "$TARGET/actor" ]; then
    # Raw verl checkpoint: verl's merger writes BASE weights to target_dir and
    # parks the adapter in lora_adapter/, so serving it directly would score the
    # base model. merge_lora.py applies the adapter and refuses to emit anything
    # unless the weights provably changed.
    TAG="${2:-$(basename "$TARGET")}"
    MODEL="$RD/merged/$(basename "$TARGET")"
    [ -d "$MODEL" ] || python "$RD/merge_lora.py" "$TARGET" "$MODEL"
else
    MODEL="$TARGET"
    TAG="${2:-$(basename "$TARGET")}"
fi

OUT="$RD/eval/$TAG"; mkdir -p "$OUT"
echo "[eval] model=$MODEL  tag=$TAG"

# VLLM_EXTRA lets a run turn off the two settings that make the forward pass
# depend on what else was in flight. Measured (base model, greedy, 522 val rows,
# two runs): 70 tasks (13.4%) flipped outcome between runs, which is NOT better
# than the 12.3% seen between two sampled runs -- greedy alone does not stabilise
# an agentic trajectory. Candidates for the remaining nondeterminism:
#   --no-enable-prefix-caching  a cache hit takes a different code path from a
#                               recompute, and whether it hits depends on run history
#   --enforce-eager             cudagraphs are captured per batch-shape bucket,
#                               so different buckets run different kernels
# Concurrency itself (batch composition -> reduction order) is not addressed by
# either; only --concurrency 1 would, at 32x the wall clock.
VLLM_EXTRA="${VLLM_EXTRA:-}"
serve() {  # gpu port logfile
    resolve_parser_flags "$MODEL"
    CUDA_VISIBLE_DEVICES="$1" nohup vllm serve "$MODEL" \
        --port "$2" --max-model-len 12288 \
        --gpu-memory-utilization 0.85 --tensor-parallel-size 1 \
        "${PARSER_FLAGS[@]}" \
        $VLLM_EXTRA \
        > "$3" 2>&1 &
    echo $!
}

# Each split gets an isolated RLLM_HOME holding ONLY the registry parquet files.
# Two things are deliberately excluded: dataset.toml and data/.
# rllm/cli/eval.py:112 redirects to a "materialised" benchmark whenever
# <RLLM_HOME>/datasets/<name>/dataset.toml exists and --agent is set, and the
# local loader then reads the split from that toml (default: test) and IGNORES
# --split. Two concurrent evals sharing one home also race to create it. Result
# seen in practice: a run labelled "val" scored 558 test rows.
for s in val test; do
    H="$RD/.rllm_$s/datasets/finqa"
    rm -rf "$RD/.rllm_$s"
    mkdir -p "$H"
    cp "$HOME/.rllm/datasets/registry.json" "$RD/.rllm_$s/datasets/" 2>/dev/null || true
    cp "$HOME"/.rllm/datasets/finqa/*.parquet "$H/" 2>/dev/null || true
done

PID_VAL=$(serve 0 "$PORT_VAL"  "$OUT/vllm_val.log")
PID_TEST=$(serve 1 "$PORT_TEST" "$OUT/vllm_test.log")
trap 'kill "$PID_VAL" "$PID_TEST" 2>/dev/null || true' EXIT
echo "[eval] vllm pids: val=$PID_VAL (gpu0:$PORT_VAL)  test=$PID_TEST (gpu1:$PORT_TEST)"

wait_ready() {  # port pid name
    for _ in $(seq 1 180); do
        curl -sf "http://localhost:$1/v1/models" >/dev/null 2>&1 && { echo "[eval] $3 ready"; return 0; }
        kill -0 "$2" 2>/dev/null || { echo "[eval] $3 server died, see $OUT" >&2; return 1; }
        sleep 5
    done
    echo "[eval] $3 never became ready" >&2; return 1
}
wait_ready "$PORT_VAL"  "$PID_VAL"  val
wait_ready "$PORT_TEST" "$PID_TEST" test

run_split() {  # split port
    # Each split gets its OWN RLLM_HOME. `rllm eval` materialises the dataset
    # into <RLLM_HOME>/datasets/finqa, and two concurrent evals sharing that
    # path race: the second one overwrites the first, so a "val" run silently
    # scored 558 test rows. Isolating the home directory removes the shared
    # mutable path entirely.
    RLLM_HOME="$RD/.rllm_$1" \
    rllm eval finqa \
        --agent finqa --evaluator finqa \
        --model "$MODEL" --base-url "http://localhost:$2/v1" \
        --split "$1" --concurrency 32 \
        --sampling-params "temperature=$EVAL_TEMP,top_p=$EVAL_TOP_P,seed=$EVAL_SEED" \
        --episodes-dir "$OUT/episodes_$1" \
        --output "$OUT/$1.json" > "$OUT/eval_$1.log" 2>&1
    echo "[eval] $1 finished (rc=$?)"
}

run_split val  "$PORT_VAL"  &
RV=$!
run_split test "$PORT_TEST" &
RT=$!
wait "$RV" "$RT"

echo "EVAL_PARALLEL_DONE tag=$TAG"
for s in val test; do
    [ -f "$OUT/$s.json" ] && python - "$OUT/$s.json" "$s" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
acc = d.get("accuracy", d.get("pass@1", d.get("mean_reward")))
n = d.get("num_examples", d.get("n"))
print(f"  {sys.argv[2]:5s} accuracy={acc}  n={n}")
PY
done

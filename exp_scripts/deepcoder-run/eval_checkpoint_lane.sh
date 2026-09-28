#!/usr/bin/env bash
# Evaluate DeepCoder verl/FSDP LoRA checkpoints with exactly the base-eval
# protocol.  This is orchestration only; flow, evaluator and dataset code are
# shared with eval_base.py.
#
# v2 -- superset of eval_checkpoint_lane.sh.  It exists as a separate file only
# because two bash instances were executing the v1 script when this was written
# (bash reads a running script by byte offset, so editing it in place would have
# derailed them).  Once no lane is executing v1, move this over it so there is
# one entry point again.
#
# Specs: host16:STEP | host24:STEP | grpo24:STEP | sft16:STEP
# SFT checkpoints are flat (model_*.pt at the top level) while RL checkpoints
# nest everything under actor/.  merge_lora.py requires actor/, so a flat source
# is staged straight into <raw>/actor/ and the merger needs no change.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
GPU="${1:?usage: $0 GPU PORT checkpoint-spec [...] }"
PORT="${2:?usage: $0 GPU PORT checkpoint-spec [...] }"
shift 2

source "$ROOT/../finqa-grpo-run/env.sh"
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPU"
export RLLM_HOME="$ROOT/runtime"
export PYTHONPATH="$ROOT:/home/yanan/agents/rllm/cookbooks/deepcoder:/home/yanan/agents/rllm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

WORK=/mnt/disk1t/deepcoder-prpo-checkpoint-eval
RAW16="$WORK/raw16"
MERGED="$WORK/merged"
RESULTS="${RESULTS_DIR:-$WORK/results}"   # protocol-specific results root (e.g. results-c32)
MERGER="$ROOT/../finqa-grpo-run/merge_lora.py"
NFS24="$ROOT/checkpoints/deepcoder-prpo-b64-n1-24-forkserver"
HOST24_REMOTE_ROOT="${HOST24_REMOTE_ROOT:-}"
mkdir -p "$RAW16" "$MERGED" "$RESULTS"

cleanup_server() {
  if [[ -n "${SERVER_PGID:-}" ]]; then
    # vLLM v1 starts EngineCore as a multiprocessing child which can be
    # re-parented after the API process exits.  Kill the dedicated process
    # group, not only the API PID, or ~29 GiB remains pinned on the GPU.
    kill -9 -- "-$SERVER_PGID" 2>/dev/null || true
  elif [[ -n "${SERVER_PID:-}" ]]; then
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID:-}" ]]; then
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=
  SERVER_PGID=
}
trap cleanup_server EXIT

for SPEC in "$@"; do
  # Specs are host24:379, grpo24:576, sft16:136, etc.
  HOST="${SPEC%%:*}"
  STEP="${SPEC##*:}"
  [[ "$HOST" =~ ^(host16|host24|grpo24|sft16)$ && "$STEP" =~ ^[0-9]+$ ]] || {
    echo "invalid checkpoint spec: $SPEC" >&2; exit 2;
  }
  TAG="${HOST}_global_step_${STEP}"
  OUT="$RESULTS/$TAG"
  if [[ -s "$OUT/result.json" ]]; then
    echo "[$TAG] complete; skipping"
    continue
  fi
  mkdir -p "$OUT"

  # Resolve where this checkpoint lives and how it is laid out.
  REMOTE=""; REMOTE_ROOT=""; LAYOUT=nested; RSYNC_EXCLUDE=()
  case "$HOST" in
    host16)
      REMOTE=10.225.68.16
      REMOTE_ROOT="${HOST16_REMOTE_ROOT:-/home/yanan/.deepcoder-checkpoints/deepcoder-prpo-b64-n1-16-c64-ni0-save10-ep3}"
      ;;
    grpo24)
      REMOTE=10.225.68.24
      REMOTE_ROOT="${GRPO24_REMOTE_ROOT:-/home/yanan/.deepcoder-checkpoints/deepcoder-grpo-b16-n8-24-c64-drop-save64}"
      ;;
    sft16)
      REMOTE=10.225.68.16
      REMOTE_ROOT="${SFT16_REMOTE_ROOT:-/home/yanan/.deepcoder-checkpoints/deepcoder-self-sft-current4360-r32-e4-len18000}"
      LAYOUT=flat
      # The merger reads model shards, fsdp_config.json and huggingface/ only;
      # optimizer state is 530 MiB per checkpoint of pure transfer cost.
      RSYNC_EXCLUDE=(--exclude 'optim_world_size_*')
      ;;
    host24)
      if [[ -n "$HOST24_REMOTE_ROOT" ]]; then
        REMOTE=10.225.68.24
        REMOTE_ROOT="$HOST24_REMOTE_ROOT"
      else
        REMOTE=""   # legacy NFS path, nothing to stage
      fi
      ;;
  esac

  STAGED=0
  if [[ -z "$REMOTE" ]]; then
    RAW="$NFS24/global_step_${STEP}"
  else
    RAW="$RAW16/$TAG"
    if [[ ! -f "$RAW/actor/lora_train_meta.json" ]]; then
      TMP="${RAW}.incoming"
      rm -rf -- "$TMP"
      # A flat source becomes a nested one here; that is the whole adaptation.
      if [[ "$LAYOUT" == flat ]]; then DEST="$TMP/actor"; else DEST="$TMP"; fi
      mkdir -p "$DEST"
      rsync -a --partial --info=progress2 "${RSYNC_EXCLUDE[@]}" \
        "${REMOTE}:${REMOTE_ROOT}/global_step_${STEP}/" \
        "$DEST/" >"$OUT/stage.log" 2>&1
      # Publish atomically: a partial copy must never become $RAW.
      test -f "$TMP/actor/lora_train_meta.json"
      test -f "$TMP/actor/fsdp_config.json"
      find "$TMP/actor" -maxdepth 1 -name 'model_world_size_*.pt' -print -quit | grep -q .
      mv "$TMP" "$RAW"
    fi
    STAGED=1
  fi
  test -f "$RAW/actor/lora_train_meta.json"

  MODEL="$MERGED/$TAG"
  rm -rf -- "$MODEL"
  python "$MERGER" "$RAW" "$MODEL" >"$OUT/merge.log" 2>&1
  test -f "$MODEL/config.json"
  find "$MODEL" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q .

  cd /home/yanan/agents/rllm
  setsid vllm serve "$MODEL" --served-model-name deepcoder-checkpoint \
    --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 \
    --max-model-len 32768 --gpu-memory-utilization 0.9 --max-num-seqs "${EVAL_CONCURRENCY:-8}" \
    >"$OUT/server.log" 2>&1 &
  SERVER_PID=$!
  SERVER_PGID=$SERVER_PID
  READY=0
  for ((i=0; i<300; i++)); do
    kill -0 "$SERVER_PID" 2>/dev/null || {
      echo "[$TAG] vLLM exited before ready" >&2; exit 1;
    }
    if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      READY=1; break
    fi
    sleep 2
  done
  [[ "$READY" == 1 ]]

  # eval_base.py pins test=687, temperature=0, top_p=1, seed=1234,
  # max_tokens=16384 and grader concurrency=8.
  rm -rf -- "$OUT/eval"
  python -u "$ROOT/eval_base.py" \
    --url "http://127.0.0.1:$PORT/v1" --model deepcoder-checkpoint \
    --output "$OUT/eval" >"$OUT/eval.log" 2>&1
  test -s "$OUT/eval/result.json"
  cp "$OUT/eval/result.json" "$OUT/result.json"
  # episodes.jsonl contains the hidden tests repeated in every episode and is
  # ~16 GiB per checkpoint. Preserve it losslessly without exhausting disk.
  rm -f      "$OUT/eval/episodes.jsonl"

  cleanup_server
  rm -rf -- "$MODEL"
  if [[ "$STAGED" == 1 ]]; then rm -rf -- "$RAW"; fi
  echo "[$TAG] COMPLETE $(tr '\n' ' ' < "$OUT/result.json")"
done

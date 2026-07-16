#!/usr/bin/env bash
# Pod smoke: bring up vLLM (Qwen3.5-9B) + gateway (responses adapter +
# cumulative token mode) + tiny multimodal_codex training run.
#
# Prereqs (Koala debug pod):
#   - HF cache with Qwen/Qwen3.5-9B (script does NOT re-download)
#   - /tmp/uv-venv exists (rLLM venv)
#   - AWS creds injected (koala auto-injects; SSH shells need manual import)
#
# Usage:  bash cookbooks/multimodal_codex/train_smoke.sh
set -euo pipefail

: "${PROJECT_DIR:=/data/work/rllm}"
: "${HF_HOME:=/data/work/hf_cache}"
: "${MODEL:=Qwen/Qwen3.5-9B}"
: "${VLLM_PORT:=4000}"
: "${GATEWAY_PORT:=8080}"

export UV_FROZEN=1
export HF_HOME
export HF_HUB_DISABLE_XET=1
# vLLM 0.22.1 links against CUDA 13; venv default is CUDA 12 — inject 13 first.
export LD_LIBRARY_PATH="/tmp/uv-venv/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_DIR"

log_dir="/local-ssd/mmcodex-smoke-$(basename "$PROJECT_DIR")"
mkdir -p "$log_dir"

echo "[smoke] starting vLLM on port $VLLM_PORT (log: $log_dir/vllm.log)"
uv run vllm serve "$MODEL" \
    --port "$VLLM_PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5 \
    --dtype bfloat16 \
    --trust-remote-code \
    --gdn-prefill-backend triton \
    > "$log_dir/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true; kill ${GATEWAY_PID:-} 2>/dev/null || true' EXIT

echo "[smoke] waiting for vLLM..."
for _ in $(seq 1 120); do
    if curl -s "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
        echo "[smoke] vLLM up"
        break
    fi
    sleep 2
done

echo "[smoke] starting gateway on port $GATEWAY_PORT (log: $log_dir/gateway.log)"
RLLM_API_FORMAT=responses \
    uv run python -m rllm_model_gateway.server \
    --cumulative-token-mode \
    --renderer-family qwen3.5 \
    --model "$MODEL" \
    --port "$GATEWAY_PORT" \
    --worker "http://localhost:$VLLM_PORT/v1" \
    > "$log_dir/gateway.log" 2>&1 &
GATEWAY_PID=$!

echo "[smoke] waiting for gateway..."
for _ in $(seq 1 60); do
    if curl -s "http://localhost:$GATEWAY_PORT/v1/models" >/dev/null 2>&1; then
        echo "[smoke] gateway up"
        break
    fi
    sleep 2
done

echo "[smoke] running training smoke (log: $log_dir/train.log)"
uv run python cookbooks/multimodal_codex/train.py \
    rllm.backend=verl \
    model.name="$MODEL" \
    training.batch_size=1 \
    training.group_size=1 \
    training.max_steps=3 \
    2>&1 | tee "$log_dir/train.log"

echo "[smoke] done — logs at $log_dir"

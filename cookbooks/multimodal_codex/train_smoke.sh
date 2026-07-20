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
# We bypass `uv run` for vLLM / gateway / train.py because `uv run` re-syncs
# the venv against the lockfile every launch, which UNDOES any manual torch
# cu128 override needed to match this pod's CUDA toolkit. Use the venv's
# python/entry-points directly.
VENV_PY=/tmp/uv-venv/bin/python
VENV_VLLM=/tmp/uv-venv/bin/vllm
# vLLM 0.22.1 links against CUDA 13; venv default is CUDA 12 — inject 13 first.
# Detect the venv's python site-packages dir (pod is py3.12; older tars used py3.13).
_VENV_SITE=$(ls -d /tmp/uv-venv/lib/python*/site-packages 2>/dev/null | head -1)
if [[ -n "$_VENV_SITE" && -d "$_VENV_SITE/nvidia/cu13/lib" ]]; then
    export LD_LIBRARY_PATH="$_VENV_SITE/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
fi

cd "$PROJECT_DIR"

log_dir="/local-ssd/mmcodex-smoke-$(basename "$PROJECT_DIR")"
mkdir -p "$log_dir"

echo "[smoke] starting vLLM on port $VLLM_PORT (log: $log_dir/vllm.log)"
"$VENV_VLLM" serve "$MODEL" \
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
    "$VENV_PY" -m rllm_model_gateway.server \
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
# 1-GPU 9B smoke config, adapted from geo3k/train_verl.sh with FSDP offload +
# small batch/prompt sizes so the rollout + a single gradient step fits on
# one H200. The goal is validating Turn 1 chat/completions round-trip through
# the gateway, not model convergence — save/test freq are set high so we
# don't checkpoint or evaluate during the smoke.
"$VENV_PY" cookbooks/multimodal_codex/train.py \
    rllm/backend=verl \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    rllm.algorithm.use_rllm=true \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    +model.name="$MODEL" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.logger="['console']" \
    trainer.project_name=mmcodex-smoke \
    trainer.experiment_name=qwen3.5-9b-1gpu \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=999 \
    trainer.test_freq=999 \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=disable \
    2>&1 | tee "$log_dir/train.log"

echo "[smoke] done — logs at $log_dir"

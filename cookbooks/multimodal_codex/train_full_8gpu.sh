#!/usr/bin/env bash
# 8-GPU normal-pod formal training for mm_codex.
#
# Layout (single H200 node × 8 GPU):
#   GPU 0-1 : External vLLM (tp=2) — gateway's OpenAI worker for CodexHarness (probe only)
#   GPU 2-7 : verl actor (6 GPU, tp=2) — training + verl-internal rollout
#
# Differs from train_smoke_8gpu.sh in the training block (Step 10):
#   - total_epochs=3           (was 1)
#   - save_freq=10             (was 999)
#   - test_freq=20             (was 999)
#   - val_before_train=true    (was false)
#   - val_batch_size=8         (was 6 — full test set)
#   - experiment_name qwen3.5-9b-grpo-r5-8gpu-v1
#   - project_name mmcodex-prod
#   - trap ADDs periodic checkpoint S3 sync so long-running progress isn't lost
#
# All other setup (venv restore, apt install, HF cache, gateway) is identical
# to the smoke script and idempotent — this can also bootstrap a fresh pod.
#
# Logs: /local-ssd/mmcodex-8gpu-full/{vllm,gateway,train}.log
# Checkpoints: /local-ssd/mmcodex-8gpu-full/exp/  → S3 sync every 60s in background
set -euo pipefail

: "${PROJECT_DIR:=/data/work/rllm}"
: "${MODEL:=Qwen/Qwen3.5-9B}"
: "${VLLM_PORT:=4000}"
: "${GATEWAY_PORT:=8080}"
: "${LOG_ROOT:=/local-ssd/mmcodex-8gpu-full}"
: "${CKPT_S3:=s3://arcwm-code-us-west-2/ericzyma/experiments/mmcodex-qwen3.5-9b-grpo-r5-v1}"

mkdir -p "$LOG_ROOT"
cd "$PROJECT_DIR"

# --------- 0. Import AWS creds if SSH shell (no-op if PID 1 already has them) ---------
if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
    eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(AWS_|KOALA_|S3_)' | sed 's/^/export /')"
fi

# --------- 1. venv restore (skip if already present) ---------
if [[ ! -x /tmp/uv-venv/bin/python ]]; then
    echo "[full] restoring rllm-venv-cu128 from S3"
    rm -rf /tmp/uv-venv
    s5cmd cat s3://arcwm-code-us-west-2/ericzyma/tools/rllm-venv-cu128.tar | tar xf - -C /tmp/
fi

# --------- 2. Fix torch cu128 (tar name lies, packs cu130) ---------
_TORCH_VERSION=$(/tmp/uv-venv/bin/python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
if [[ "$_TORCH_VERSION" != *"cu128"* ]]; then
    echo "[full] reinstalling torch==2.11.0+cu128 (was: $_TORCH_VERSION)"
    /tmp/uv-venv/bin/uv pip install --python /tmp/uv-venv/bin/python \
        --reinstall torch==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu128
fi

# --------- 3. Apt packages ---------
if ! command -v bwrap >/dev/null 2>&1; then
    echo "[full] installing apt packages"
    apt-get update -qq
    apt-get install -y -qq bubblewrap ninja-build
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1
    apt-get install -y -qq nodejs
    npm install -g @openai/codex >/dev/null 2>&1
fi

# --------- 4. HF cache ---------
export HF_HOME=/local-ssd/hf_cache
mkdir -p "$HF_HOME" /local-ssd/rllm_home
if [[ ! -d "$HF_HOME/hub/models--Qwen--Qwen3.5-9B" ]]; then
    echo "[full] restoring HF cache for Qwen3.5-9B"
    s5cmd cat s3://arcwm-code-us-west-2/ericzyma/tools/hf_cache_qwen3.5-9b.tar | tar xf - -C /local-ssd/
fi

# --------- 5. Environment ---------
export UV_FROZEN=1
export HF_HUB_DISABLE_XET=1
export RLLM_HOME=/local-ssd/rllm_home
export PATH="/tmp/uv-venv/bin:${PATH}"
export RLLM_API_FORMAT=responses  # Codex CLI speaks Responses API — needed for both external + verl-internal gateway
_VENV_SITE=$(ls -d /tmp/uv-venv/lib/python*/site-packages 2>/dev/null | head -1)
if [[ -n "$_VENV_SITE" && -d "$_VENV_SITE/nvidia/cu13/lib" ]]; then
    export LD_LIBRARY_PATH="$_VENV_SITE/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
fi

VENV_PY=/tmp/uv-venv/bin/python
VENV_VLLM=/tmp/uv-venv/bin/vllm

# --------- 6. Prepare data ---------
echo "[full] preparing data"
"$VENV_PY" -m cookbooks.multimodal_codex.prepare_data

# --------- 7. External vLLM on GPU 0-1 ---------
echo "[full] starting external vLLM on GPU 0-1 (log: $LOG_ROOT/vllm.log)"
CUDA_VISIBLE_DEVICES=0,1 "$VENV_VLLM" serve "$MODEL" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.7 \
    --dtype bfloat16 \
    --trust-remote-code \
    --gdn-prefill-backend triton \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    > "$LOG_ROOT/vllm.log" 2>&1 &
VLLM_PID=$!

# Cleanup + last-ditch checkpoint sync on exit
CKPT_SYNC_PID=""
trap '
    kill $VLLM_PID 2>/dev/null || true
    kill ${GATEWAY_PID:-} 2>/dev/null || true
    kill ${CKPT_SYNC_PID:-} 2>/dev/null || true
    if [[ -d "$LOG_ROOT/exp" ]]; then
        echo "[full] final checkpoint sync to $CKPT_S3/"
        aws s3 sync "$LOG_ROOT/exp/" "$CKPT_S3/exp/" --quiet || true
        aws s3 sync "$LOG_ROOT/" "$CKPT_S3/logs/" --exclude "exp/*" --quiet || true
    fi
' EXIT

echo "[full] waiting for vLLM (up to 600s)"
for _ in $(seq 1 300); do
    if curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
        echo "[full] vLLM up"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[full] FATAL: vLLM died during startup"
        tail -80 "$LOG_ROOT/vllm.log"
        exit 1
    fi
    sleep 2
done

# --------- 8. Gateway ---------
echo "[full] starting gateway on port $GATEWAY_PORT"
"$VENV_PY" -m rllm_model_gateway.server \
    --cumulative-token-mode \
    --renderer-family qwen3.5 \
    --model "$MODEL" \
    --port "$GATEWAY_PORT" \
    --worker "http://localhost:$VLLM_PORT/v1" \
    > "$LOG_ROOT/gateway.log" 2>&1 &
GATEWAY_PID=$!

echo "[full] waiting for gateway"
for _ in $(seq 1 60); do
    if curl -sf "http://localhost:$GATEWAY_PORT/v1/models" >/dev/null 2>&1; then
        echo "[full] gateway up"
        break
    fi
    if ! kill -0 $GATEWAY_PID 2>/dev/null; then
        echo "[full] FATAL: gateway died"
        tail -80 "$LOG_ROOT/gateway.log"
        exit 1
    fi
    sleep 2
done

# --------- 9. Sanity probe ---------
echo "[full] running probe_rollout to verify e2e path"
if "$VENV_PY" cookbooks/multimodal_codex/probe_rollout.py 2>&1 | tee "$LOG_ROOT/probe.log"; then
    echo "[full] probe passed"
else
    echo "[full] WARN: probe failed (proceeding to training anyway)"
fi

# --------- 9.5. Background checkpoint S3 sync (every 60s) ---------
(
    while true; do
        sleep 60
        if [[ -d "$LOG_ROOT/exp" ]]; then
            aws s3 sync "$LOG_ROOT/exp/" "$CKPT_S3/exp/" --quiet || true
        fi
    done
) &
CKPT_SYNC_PID=$!
echo "[full] background checkpoint sync started (pid=$CKPT_SYNC_PID → $CKPT_S3)"

# --------- 10. verl formal training on GPU 2-7 ---------
echo "[full] starting verl training on GPU 2-7 (log: $LOG_ROOT/train.log)"
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 "$VENV_PY" cookbooks/multimodal_codex/train.py \
    rllm/backend=verl \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    rllm.algorithm.use_rllm=true \
    data.train_batch_size=6 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    +model.name="$MODEL" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=30 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=5120 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=qwen3_coder \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.logger="['console']" \
    trainer.project_name=mmcodex-prod \
    trainer.experiment_name=qwen3.5-9b-grpo-r5-v1 \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="$LOG_ROOT/exp" \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=auto \
    2>&1 | tee "$LOG_ROOT/train.log"

echo "[full] training done — logs at $LOG_ROOT, checkpoints at $CKPT_S3"

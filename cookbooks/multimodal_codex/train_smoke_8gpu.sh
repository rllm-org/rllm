#!/usr/bin/env bash
# 8-GPU normal-pod smoke for mm_codex.
#
# Layout (single H200 node × 8 GPU):
#   GPU 0-1 : External vLLM (tp=2) — gateway's OpenAI worker for CodexHarness
#   GPU 2-7 : verl actor (6 GPU, tp=2) — training + verl-internal rollout
#
# Contract:
#   - Pod submitted with `koala submit -m normal -n 1 -g 8 --large-ssd --s3-log`
#   - Runs venv restore + apt setup + HF cache prefetch before smoke, so it can
#     also serve as first-time bootstrap on a fresh pod.
#   - Meant for smoke: 1 training step, val_before_train=false, save/test off.
#
# Logs: /local-ssd/mmcodex-8gpu-smoke/{vllm,gateway,train}.log
# Koala also mirrors stdout/stderr to s3://.../.koala-logs/$JOB_NAME/ via --s3-log.
set -euo pipefail

: "${PROJECT_DIR:=/data/work/rllm}"
: "${MODEL:=Qwen/Qwen3.5-9B}"
: "${VLLM_PORT:=4000}"
: "${GATEWAY_PORT:=8080}"
: "${LOG_ROOT:=/local-ssd/mmcodex-8gpu-smoke}"

mkdir -p "$LOG_ROOT"
cd "$PROJECT_DIR"

# --------- 0. Import AWS creds if SSH shell (no-op if PID 1 already has them) ---------
if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
    eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(AWS_|KOALA_|S3_)' | sed 's/^/export /')"
fi

# --------- 1. venv restore (skip if already present) ---------
if [[ ! -x /tmp/uv-venv/bin/python ]]; then
    echo "[smoke] restoring rllm-venv-cu128 from S3"
    rm -rf /tmp/uv-venv
    s5cmd cat s3://arcwm-code-us-west-2/ericzyma/tools/rllm-venv-cu128.tar | tar xf - -C /tmp/
fi

# --------- 2. Fix torch cu128 (tar name lies, packs cu130) ---------
_TORCH_VERSION=$(/tmp/uv-venv/bin/python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
if [[ "$_TORCH_VERSION" != *"cu128"* ]]; then
    echo "[smoke] reinstalling torch==2.11.0+cu128 (was: $_TORCH_VERSION)"
    /tmp/uv-venv/bin/uv pip install --python /tmp/uv-venv/bin/python \
        --reinstall torch==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu128
fi

# --------- 3. Apt packages (bubblewrap for sandbox, ninja for torch.compile, node for codex CLI) ---------
if ! command -v bwrap >/dev/null 2>&1; then
    echo "[smoke] installing apt packages"
    apt-get update -qq
    apt-get install -y -qq bubblewrap ninja-build
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1
    apt-get install -y -qq nodejs
    npm install -g @openai/codex >/dev/null 2>&1
fi

# --------- 4. HF cache (skip if Qwen3.5-9B present) ---------
export HF_HOME=/local-ssd/hf_cache
mkdir -p "$HF_HOME" /local-ssd/rllm_home
if [[ ! -d "$HF_HOME/hub/models--Qwen--Qwen3.5-9B" ]]; then
    echo "[smoke] restoring HF cache for Qwen3.5-9B"
    s5cmd cat s3://arcwm-code-us-west-2/ericzyma/tools/hf_cache_qwen3.5-9b.tar | tar xf - -C /local-ssd/
fi

# --------- 5. Environment ---------
export UV_FROZEN=1
export HF_HUB_DISABLE_XET=1
export RLLM_HOME=/local-ssd/rllm_home
export PATH="/tmp/uv-venv/bin:${PATH}"
_VENV_SITE=$(ls -d /tmp/uv-venv/lib/python*/site-packages 2>/dev/null | head -1)
if [[ -n "$_VENV_SITE" && -d "$_VENV_SITE/nvidia/cu13/lib" ]]; then
    export LD_LIBRARY_PATH="$_VENV_SITE/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
fi

VENV_PY=/tmp/uv-venv/bin/python
VENV_VLLM=/tmp/uv-venv/bin/vllm

# --------- 6. Prepare data (idempotent — DatasetRegistry caches on disk) ---------
echo "[smoke] preparing data"
"$VENV_PY" -m cookbooks.multimodal_codex.prepare_data

# --------- 7. External vLLM on GPU 0-1 (tp=2, gateway backend) ---------
echo "[smoke] starting external vLLM on GPU 0-1 (log: $LOG_ROOT/vllm.log)"
CUDA_VISIBLE_DEVICES=0,1 "$VENV_VLLM" serve "$MODEL" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.7 \
    --dtype bfloat16 \
    --trust-remote-code \
    --gdn-prefill-backend triton \
    > "$LOG_ROOT/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true; kill ${GATEWAY_PID:-} 2>/dev/null || true' EXIT

echo "[smoke] waiting for vLLM (up to 600s)"
for _ in $(seq 1 300); do
    if curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
        echo "[smoke] vLLM up"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[smoke] FATAL: vLLM died during startup"
        tail -80 "$LOG_ROOT/vllm.log"
        exit 1
    fi
    sleep 2
done

# --------- 8. Gateway (harness OpenAI API interceptor) ---------
echo "[smoke] starting gateway on port $GATEWAY_PORT"
RLLM_API_FORMAT=responses \
    "$VENV_PY" -m rllm_model_gateway.server \
    --cumulative-token-mode \
    --renderer-family qwen3.5 \
    --model "$MODEL" \
    --port "$GATEWAY_PORT" \
    --worker "http://localhost:$VLLM_PORT/v1" \
    > "$LOG_ROOT/gateway.log" 2>&1 &
GATEWAY_PID=$!

echo "[smoke] waiting for gateway"
for _ in $(seq 1 60); do
    if curl -sf "http://localhost:$GATEWAY_PORT/v1/models" >/dev/null 2>&1; then
        echo "[smoke] gateway up"
        break
    fi
    if ! kill -0 $GATEWAY_PID 2>/dev/null; then
        echo "[smoke] FATAL: gateway died"
        tail -80 "$LOG_ROOT/gateway.log"
        exit 1
    fi
    sleep 2
done

# [MMCODEX-DIAG] surface gateway startup markers to stdout so koala S3 log
# captures them (gateway.log itself is only inside /local-ssd/).
echo "[smoke] === gateway startup markers ==="
grep -E "MMCODEX-DIAG|RLLM_API_FORMAT" "$LOG_ROOT/gateway.log" | head -20 || echo "[smoke] (no MMCODEX-DIAG lines in gateway.log)"
echo "[smoke] === end gateway startup markers ==="

# --------- 9. Sanity probe (D-path: gateway → vLLM → image → answer) ---------
echo "[smoke] running probe_rollout to verify e2e path"
if "$VENV_PY" cookbooks/multimodal_codex/probe_rollout.py 2>&1 | tee "$LOG_ROOT/probe.log"; then
    echo "[smoke] probe passed"
else
    echo "[smoke] WARN: probe failed (may still be OK for training smoke)"
fi

# --------- 10. verl smoke training on GPU 2-7 (6 GPU, tp=2) ---------
echo "[smoke] starting verl training on GPU 2-7 (log: $LOG_ROOT/train.log)"
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 "$VENV_PY" cookbooks/multimodal_codex/train.py \
    rllm/backend=verl \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    rllm.algorithm.use_rllm=true \
    data.train_batch_size=6 \
    data.val_batch_size=6 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    +model.name="$MODEL" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
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
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=5120 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.logger="['console']" \
    trainer.project_name=mmcodex-smoke \
    trainer.experiment_name=qwen3.5-9b-8gpu \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.save_freq=999 \
    trainer.test_freq=999 \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=disable \
    2>&1 | tee "$LOG_ROOT/train.log"

echo "[smoke] done — logs at $LOG_ROOT"

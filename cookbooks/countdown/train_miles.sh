#!/usr/bin/env bash
# Countdown via the AgentFlow protocol (@rllm.rollout + @rllm.evaluator) on the Miles
# backend: SGLang rollout + FSDP2 training, 4 train GPUs + 4 rollout GPUs on one 8-GPU node.
#
# This is the AgentFlow counterpart to
# examples/countdown/unified_trainer/train_countdown_unified_miles.sh (SimpleWorkflow).
# The difference matters: AgentFlow agents talk OpenAI over rLLM's *gateway*, which
# registers Miles' SGLang router as an inference worker and reconstructs tokens from the
# captured traces. SimpleWorkflow instead goes straight through MilesEngine's
# token-in/token-out path. Same trainer, different rollout plumbing.
#
# Prerequisites:
#   1. bash scripts/setup_miles_env.sh          (or an existing miles venv)
#   2. python cookbooks/countdown/prepare_data.py
#   3. uv pip install --no-deps -e cookbooks/countdown
#
# Usage:
#   PYTHON=/path/to/miles-venv/bin/python bash cookbooks/countdown/train_miles.sh [overrides...]
#
# Any trailing argument is a Hydra override, e.g.
#   ... bash cookbooks/countdown/train_miles.sh rllm.trainer.total_batches=60
set -x
set -euo pipefail

PY="${PYTHON:-python}"
PYBIN="$(cd "$(dirname "$(command -v "$PY")")" && pwd)"

# SGLang JIT-compiles kernels at engine start, so ninja and nvcc must be on PATH in the
# Ray workers (they inherit the launching shell's PATH).
export PATH="${PYBIN}:/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# Triton's cache must be a FRESH dir on tmpfs: a stale ~/.triton/cache, or one on an
# overlay fs, makes concurrent ranks race during the first kernel compile and surfaces as
# "__triton_launcher...so: cannot open shared object file".
export TRITON_CACHE_DIR="$(mktemp -d /dev/shm/triton.XXXXXX)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
trap 'rm -rf "$TRITON_CACHE_DIR"' EXIT

# The model is already cached, and a run fires ~8 concurrent Hub lookups (4 train actors
# + 4 SGLang engines) which reliably trips HF rate limiting. Set HF_TOKEN instead if you
# need to pull a new model.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cd "$(dirname "${BASH_SOURCE[0]}")"

"$PY" -u train.py \
    rllm/backend=miles \
    model.name=Qwen/Qwen3-1.7B \
    miles.train_backend=fsdp \
    miles.actor_num_nodes=1 \
    miles.actor_num_gpus_per_node=4 \
    miles.rollout_num_gpus=4 \
    miles.rollout_num_gpus_per_engine=1 \
    miles.global_batch_size=128 \
    miles.max_tokens_per_gpu=8192 \
    miles.lr=2e-6 \
    rllm.data.train_batch_size=16 \
    rllm.data.max_prompt_length=512 \
    rllm.data.max_response_length=1024 \
    rllm.rollout.n=8 \
    rllm.rollout.n_val=1 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.disable_thinking=true \
    rllm.algorithm.adv_estimator=grpo \
    rllm.workflow.n_parallel_tasks=64 \
    rllm.trainer.logger=['console'] \
    rllm.trainer.project_name='rllm-countdown-miles' \
    rllm.trainer.experiment_name='countdown-agentflow-miles' \
    rllm.trainer.total_batches=20 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=10 \
    rllm.trainer.save_freq=1000 \
    "$@"

# Sanity checks on the output, in order of what has actually bitten this backend:
#   train/tis_abs        ~0.01   - trainer and SGLang agree. Anything larger and the
#                                 forward passes diverge; nothing downstream is valid.
#   batch/miles_samples  > 0     - the transform produced trainable sequences
#   train/loss, grad_norm        - present at all means Miles' optimizer metrics arrived

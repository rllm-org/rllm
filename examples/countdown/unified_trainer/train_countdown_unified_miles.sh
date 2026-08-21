#!/usr/bin/env bash
# Countdown on the Miles backend (SGLang rollout + FSDP2 training), 8 GPUs.
#
# Smoke-test shape: Qwen3-1.7B, 4 train GPUs + 4 rollout GPUs, thinking disabled so
# responses stay short, 8 steps. Meant to exercise the whole loop quickly, not to reach a
# good score. Raise rllm.trainer.total_batches and the model size for real runs.
#
# Prerequisites:
#   1. miles installed (see design/miles-training-backend.md §0.1)
#   2. python examples/countdown/prepare_countdown_data.py
set -x

# SGLang JIT-compiles some kernels at engine start, so `ninja` and `nvcc` must be on
# PATH in the Ray workers (they inherit the launching shell's PATH). ninja comes from
# the venv; nvcc from the system CUDA toolkit, which must match the torch CUDA build.
PYBIN="$(cd "$(dirname "$(command -v "${PYTHON:-python}")")" && pwd)"
export PATH="${PYBIN}:/usr/local/cuda/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
nvcc --version | tail -1
ninja --version

# Triton's cache must be a FRESH dir on tmpfs: a stale ~/.triton/cache, or one on
# an overlay fs, makes concurrent ranks race during the first kernel compile and
# surfaces as "__triton_launcher...so: cannot open shared object file".
export TRITON_CACHE_DIR="$(mktemp -d /dev/shm/triton.XXXXXX)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
trap 'rm -rf "$TRITON_CACHE_DIR"' EXIT

/home/sijuntan/.claude/jobs/9cad83b2/tmp/mvenv/bin/python -m examples.countdown.unified_trainer.train_countdown_unified_miles \
    rllm/backend=miles \
    model.name=Qwen/Qwen3-1.7B \
    miles.train_backend=fsdp \
    miles.actor_num_nodes=1 \
    miles.actor_num_gpus_per_node=4 \
    miles.rollout_num_gpus=4 \
    miles.rollout_num_gpus_per_engine=1 \
    miles.global_batch_size=64 \
    miles.max_tokens_per_gpu=16384 \
    miles.lr=1e-6 \
    rllm.data.train_batch_size=8 \
    rllm.data.max_prompt_length=512 \
    rllm.data.max_response_length=1024 \
    rllm.rollout.n=8 \
    rllm.rollout.n_val=1 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    rllm.algorithm.adv_estimator=grpo \
    rllm.disable_thinking=true \
    rllm.trainer.logger=['console'] \
    rllm.trainer.project_name='rllm-countdown-miles' \
    rllm.trainer.experiment_name='smoke-qwen3-0.6b' \
    rllm.trainer.total_batches=8 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.val_before_train=false \
    rllm.trainer.test_freq=1000 \
    rllm.trainer.save_freq=1000

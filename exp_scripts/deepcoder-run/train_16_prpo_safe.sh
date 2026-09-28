#!/usr/bin/env bash
# DeepCoder PRPO on .16: CPU-grader-safe protocol selected by repeated A/B probes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export RAY_worker_niceness=0
export DEEPCODER_HOST=16
export DEEPCODER_EPOCHS=3
# PRPO consumes 64 distinct training tasks per optimizer step.  Save every
# 10 steps = 640 tasks.  Online validation remains disabled because rLLM's
# trainer validation and our offline run_dataset evaluator are separate code
# paths (and the trainer path does not expose the same explicit seed contract).
export DEEPCODER_SAVE_FREQ=10
export DEEPCODER_PARALLEL_TASKS=64
export DEEPCODER_RUN_NAME=deepcoder-prpo-b64-n1-16-c64-ni0-save10-ep3
# `.16` has a writable 1.1-TiB local ext4 filesystem at /home/yanan;
# /mnt/disk1t on that host is an unrelated, root-owned 98-GiB filesystem.
export DEEPCODER_CHECKPOINT_ROOT=/home/yanan/.deepcoder-checkpoints
exec bash "$ROOT/train.sh" prpo "$@"

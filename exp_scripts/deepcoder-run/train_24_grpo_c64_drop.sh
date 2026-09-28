#!/usr/bin/env bash
# DeepCoder GRPO on .24: full concurrency (n_parallel_tasks=64, both GPUs serve rollouts),
# with verifier timeouts DROPPED from the GRPO group instead of scored 0.
#
# Why: on .24 the CPU grader stalls under concurrency and 3-6% of rollouts were judged
# "Time Limit Exceeded" independent of the submitted code (measured 2026-09-17). Lowering
# n_parallel_tasks to 4 / 1 cut the rate to ~1% / ~0.1% but cost 4x / 12x wall time and,
# at 1, left GPU1 idle. Here compatible_grader raises GraderTimeout on a verifier timeout
# (DEEPCODER_DROP_TIMEOUTS=1) -> TerminationReason.ERROR -> rllm.compact_filtering
# (mask_error=True) excludes that rollout. The .16 pipelines keep the default (no drop).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export RAY_worker_niceness=0
# .24 host issue: NCCL peer-to-peer must be disabled or FSDP/NCCL init hangs at 'Before FSDP'
# (both ranks spin at ~100% CPU/GPU, 5.6 GB, no progress). train.sh only sets this in its prpo
# branch; the grpo branch does not, so the GRPO launcher must export it explicitly.
export NCCL_P2P_DISABLE=1
export DEEPCODER_HOST=24
export DEEPCODER_EPOCHS=30
export DEEPCODER_SAVE_FREQ=64            # 64 steps x 16 tasks = 1,024 tasks per checkpoint
export DEEPCODER_PARALLEL_TASKS=64
export DEEPCODER_DROP_TIMEOUTS=1
export DEEPCODER_RUN_NAME=deepcoder-grpo-b16-n8-24-c64-drop-save64
export DEEPCODER_CHECKPOINT_ROOT=/home/yanan/.deepcoder-checkpoints   # .24 local ext4
# raise_on_error=false is REQUIRED with timeout dropping: after retry_limit (3) consecutive
# GraderTimeouts on one rollout the engine otherwise re-raises and the whole run aborts
# (observed at step 5, 2026-09-18 17:53). With false it returns a TerminationReason.ERROR
# episode, which compact_filtering drops. Key exists in the generated schema (default True), so a plain override is used.
exec bash "$ROOT/train.sh" grpo \
  rllm.compact_filtering.enable=true \
  rllm.compact_filtering.mask_error=true \
  rllm.workflow.raise_on_error=false \
  trainer.resume_mode=auto \
  "$@"
# trainer.resume_mode=auto: train.sh hard-codes resume_mode=disable; this later override wins
# (verified with --cfg job) so a crash resumes from latest_checkpointed_iteration.txt in the
# same default_local_dir instead of restarting from step 0 (2026-09-19: host OOM at step 233,
# resumed from 192).

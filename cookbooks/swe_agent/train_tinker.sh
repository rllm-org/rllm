#!/usr/bin/env bash
# Train mini-swe-agent on swesmith, validate on swebench-verified, Tinker backend.
#
# Prerequisites:
#   1. Install rllm with tinker extras:    uv pip install -e ".[tinker]"
#   2. Pull both harbor datasets:
#        rllm dataset pull harbor:swesmith
#        rllm dataset pull harbor:swebench-verified
#   3. Docker Desktop running (sandboxed agents pull per-task swebench images).

set -euo pipefail

# Per-rollout sandbox setup + verifier is heavy (~10-15 min/task on swesmith,
# ~13-18 min/task on swebench-verified). Validation cadence + val dataset size
# dominate wall-clock time — see README's "Cost knobs".

python -u train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-30B-A3B \
    model.lora_rank=32 \
    training.group_size=4 \
    data.train_batch_size=2 \
    data.val_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=20 \
    rllm.trainer.project_name=swe_agent \
    rllm.trainer.experiment_name=mini-swe-agent_qwen3-30b-a3b \
    rllm.trainer.logger=[console,ui] \
    "$@"

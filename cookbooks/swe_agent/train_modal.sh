#!/usr/bin/env bash
# Train mini-swe-agent on swesmith with Modal sandboxes (faster than local Docker).
#
# Prerequisites:
#   1. Install rllm with tinker extras:    uv pip install -e ".[tinker]"
#   2. Pull harbor datasets:
#        rllm dataset pull harbor:swesmith
#        rllm dataset pull harbor:swebench-verified
#   3. Modal auth:                          modal setup     (or set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET)
#   4. Cloudflared (for the gateway tunnel — Modal sandboxes can't reach localhost):
#        brew install cloudflared            (macOS)
#        # or download: https://developers.cloudflare.com/cloudflare-tunnel/downloads/
#
# What's different vs train_tinker.sh:
#   --sandbox-backend=modal              run task containers in Modal's cloud
#   --sandbox-concurrency=64             raise per-flow ceiling (Modal scales out;
#                                        the harness's local-docker default is 4)
#   AgentTrainer auto-spawns a cloudflared tunnel to expose the gateway publicly
#   (Modal sandboxes call back to it for LLM completions). Override with
#   ``rllm.gateway.public_url=https://your.host`` if you have a fixed URL.

set -euo pipefail

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
    rllm.trainer.experiment_name=mini-swe-agent_qwen3-30b-a3b_modal \
    rllm.trainer.logger=[console,ui] \
    +rllm.sandbox_backend=modal \
    +rllm.sandbox_concurrency=64 \
    "$@"

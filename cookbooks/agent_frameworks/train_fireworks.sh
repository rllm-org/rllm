#!/usr/bin/env bash
# Train one of the framework AgentFlows on the Fireworks backend.
#
# Usage:
#   bash cookbooks/agent_frameworks/train_fireworks.sh                       # default: langgraph_math
#   bash cookbooks/agent_frameworks/train_fireworks.sh strands_math
#   bash cookbooks/agent_frameworks/train_fireworks.sh openai_agents_math
#   bash cookbooks/agent_frameworks/train_fireworks.sh smolagents_math
#
# Prerequisites:
#   1. Install rllm with fireworks extras:  uv pip install -e ".[fireworks]"
#   2. Install this cookbook + the framework you want:
#        uv pip install --no-deps -e "cookbooks/agent_frameworks[langgraph]"
#        uv pip install --no-deps -e "cookbooks/agent_frameworks[openai-agents]"
#        uv pip install --no-deps -e "cookbooks/agent_frameworks[smolagents]"
#        uv pip install --no-deps -e "cookbooks/agent_frameworks[strands]"
#        # or all four at once:
#        uv pip install --no-deps -e "cookbooks/agent_frameworks[all]"
#   3. Pull the datasets:                rllm dataset pull deepscaler_math && rllm dataset pull math500
#   4. Set your API key:                 export FIREWORKS_API_KEY=...
#
# The trainer job and inference deployment are provisioned on Fireworks at
# startup and torn down on shutdown.

set -euo pipefail

AGENT="${1:-langgraph_math}"
shift || true

python -u train.py \
    +rllm.agent_name=$AGENT \
    rllm/backend=fireworks \
    model.name=accounts/fireworks/models/qwen3-4b-instruct-2507 \
    model.tokenizer_model=Qwen/Qwen3-4B-Instruct-2507 \
    model.lora_rank=32 \
    fireworks_config.policy_trainer_shape_id=accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora \
    training.group_size=8 \
    data.train_batch_size=32 \
    data.val_batch_size=500 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    rllm.trainer.total_epochs=1 \
    rllm.trainer.test_freq=10 \
    rllm.trainer.project_name=agent_frameworks \
    rllm.trainer.experiment_name=$AGENT-fireworks \
    rllm.trainer.logger=[console,ui] \
    "$@"

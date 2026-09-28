#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/../finqa-grpo-run/env.sh"
source "$VENV/bin/activate"
export RLLM_HOME="$ROOT/runtime"
export PYTHONPATH="/home/yanan/agents/rllm/cookbooks/deepcoder:/home/yanan/agents/rllm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd /home/yanan/agents/rllm
exec python -u "$ROOT/prepare_full.py"

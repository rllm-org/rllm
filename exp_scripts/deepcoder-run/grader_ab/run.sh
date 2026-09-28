#!/usr/bin/env bash
# Low-priority, CPU-only, read-only probe. Never touches GPUs or the training processes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:/home/yanan/agents/rllm/cookbooks/deepcoder:/home/yanan/agents/rllm"
export PROBE_HOST="${1:?host label}"
cd /home/yanan/agents/rllm
exec nice -n 19 /home/yanan/agents/rllm/.venv/bin/python -u "$ROOT/grader_ab/probe.py"

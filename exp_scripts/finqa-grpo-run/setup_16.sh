#!/usr/bin/env bash
# One-time .16 setup. The venv and repo come over NFS from .29 (same sm_89 arch,
# so flash-attn does not need rebuilding); only the interpreter, the HF model
# cache and ~/.rllm are local to .16.
set -euo pipefail
RD=/home/yanan/agents/rllm/exp_scripts/finqa-grpo-run
source "$RD/env.sh"
source "$VENV/bin/activate"

echo "=== [1/4] judge endpoint reachable from .16? ==="
curl -sf -m 15 -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_BASE_URL/models" \
  | python -c "import sys,json;print('  serving:', [m['id'] for m in json.load(sys.stdin)['data']])"

echo "=== [2/4] register FinQA splits in .16-local ~/.rllm ==="
cd /home/yanan/agents/rllm/cookbooks/finqa
python prepare_finqa_data.py

echo "=== [3/4] prefetch Qwen3-4B-Instruct-2507 into .16 HF cache ==="
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-4B-Instruct-2507")
print("  snapshot:", p)
PY

echo "=== [4/4] judge preflight (real _call_judge calls) ==="
python "$RD/preflight_judge.py"

echo "SETUP_16_DONE"

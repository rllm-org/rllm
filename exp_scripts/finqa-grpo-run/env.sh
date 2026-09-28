# Shared environment for the FinQA GRPO runs. ~/agents is NFS-mounted from .29,
# so this file and the venv are identical on .29 and .16 (both sm_89).
VENV=/home/yanan/agents/rllm/.venv

# vllm 0.22.1's compiled extension is built against CUDA 13 while torch here is
# +cu129; libcudart.so.13 ships inside the venv via nvidia-cutlass-dsl-libs-cu13
# but is not on the loader path. Verified by real generation (137*24 -> 3288),
# not just by a successful import.
export LD_LIBRARY_PATH=$VENV/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0

# Judge -> Azure gpt-5.4-nano (the analogue of finqa.md's own gpt-5-nano).
#
# Switched from the self-hosted Qwen3.8-27B on 2026-08-28 at step 71. Reason:
# a measured 12% of Qwen judge calls came back with
# incomplete_details.reason == "max_output_tokens", and in that state the
# Responses API returns an EMPTY output_text, which _call_judge turns
# deterministically into False -> reward 0. One-directional bias: it can only
# mark a correct answer wrong. gpt-5.4-nano showed 0/393 truncations in the
# four-way A/B, with 99.7% accuracy and strictness within 0.2 pt of Qwen.
#
# temperature is deliberately NOT set anywhere - the server default applies,
# which is also required for the gpt-5.5 / 5.6 families.
# Credentials live in a mode-600 file inside this run dir, NOT in ~/.bashrc:
# ~/.bashrc is machine-local (only ~/agents is NFS-shared), so .16 cannot see
# the Azure keys that exist on .29. Reading them from here is what makes the
# judge reachable from both hosts.
CREDS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.azure_creds"
[ -r "$CREDS" ] || { echo "judge: missing $CREDS" >&2; return 1 2>/dev/null || exit 1; }
. "$CREDS"
export FINQA_JUDGE_MODEL=gpt-5.4-nano
export FINQA_MULTI_TABLE_JUDGE_MODEL=gpt-5.4-mini

# Retry any call whose finish_reason is not "stop" (truncation, content filter,
# transport error); no backoff. Every call's finish_reason is appended here.
export FINQA_JUDGE_MAX_ATTEMPTS=10
# Allow independent comparison arms to keep judge telemetry in their own run
# directory.  The fallback preserves the historical GRPO location/semantics.
export FINQA_JUDGE_FINISH_LOG="${FINQA_JUDGE_FINISH_LOG:-/home/yanan/agents/rllm/exp_scripts/finqa-grpo-run/logs/judge_finish.tsv}"

: "${OPENAI_BASE_URL:?judge endpoint unset}"
: "${FINQA_JUDGE_MODEL:?judge model unset}"

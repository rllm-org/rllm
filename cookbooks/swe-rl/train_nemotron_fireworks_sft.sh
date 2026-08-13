#!/usr/bin/env bash
# Launch the measured Nemotron 3 Nano Fireworks SFT recipe on the already
# prepared train/validation Parquet files.
#
# Usage:
#   export FIREWORKS_API_KEY=...
#   bash cookbooks/swe-rl/train_nemotron_fireworks_sft.sh [DATASET_DIR]
#
# DATASET_DIR must contain train.parquet, validation.parquet, and preferably
# manifest.json. When the files are absent, this script downloads the pinned
# private Hugging Face dataset. You may override that source explicitly:
#
#   NEMOTRON_SFT_DATASET_REPO=org/repo \
#   NEMOTRON_SFT_DATASET_REVISION=<immutable-commit> \
#   bash cookbooks/swe-rl/train_nemotron_fireworks_sft.sh /path/to/cache
#
# Set NEMOTRON_SFT_DRY_RUN=1 to verify inputs and print the command without
# provisioning a paid trainer.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/nemotron_fireworks_sft.yaml"
DEFAULT_DATASET_DIR="${REPO_ROOT}/artifacts/nemotron-swe/approx-a/migrated"
DATASET_DIR="${1:-${NEMOTRON_SFT_DATASET_DIR:-${DEFAULT_DATASET_DIR}}}"
TRAIN_PATH="${DATASET_DIR}/train.parquet"
VALIDATION_PATH="${DATASET_DIR}/validation.parquet"
MANIFEST_PATH="${DATASET_DIR}/manifest.json"
EXPECTED_TRAIN_SHA256="da467b1937cc3f2e7ce8687b8b6ea122090984faa3080becf3a3b37eb0900e16"
EXPECTED_VALIDATION_SHA256="76a18380d127d2c65716ec1c063667dfc91577280366e39702b71aa9cd9aa9e9"
EXPECTED_CONFIG_SHA256="0ad7b8a439397d725d187d10aeb9672488b7836a7edfe9141cf9c239dcebab2c"
REQUIRED_RLLM_BASE_REVISION="079c65d4237da01de0334735f3684f475c4d976f"
TOKENIZER_REVISION="2d59de1cbd51c0adf384eb906b766d1aee0e0517"
DEFAULT_DATASET_REPO="mobius-lab/nemotron-swe-v3-rllm-sft-surrogate-a"
DEFAULT_DATASET_REVISION="e775ddc45209d327afd8e26d9571c9761ca3d7ae"

if [[ $# -gt 1 ]]; then
    echo "usage: $0 [DATASET_DIR]" >&2
    exit 2
fi

if [[ ! -f "${TRAIN_PATH}" || ! -f "${VALIDATION_PATH}" ]]; then
    DATASET_REPO="${NEMOTRON_SFT_DATASET_REPO:-${DEFAULT_DATASET_REPO}}"
    DATASET_REVISION="${NEMOTRON_SFT_DATASET_REVISION:-${DEFAULT_DATASET_REVISION}}"
    if ! command -v hf >/dev/null 2>&1; then
        echo "the Hugging Face 'hf' CLI is required to download ${DATASET_REPO}" >&2
        exit 1
    fi
    mkdir -p "${DATASET_DIR}"
    hf download "${DATASET_REPO}" \
        train.parquet validation.parquet manifest.json README.md \
        --repo-type dataset \
        --revision "${DATASET_REVISION}" \
        --local-dir "${DATASET_DIR}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "missing SFT recipe: ${CONFIG_PATH}" >&2
    exit 1
fi

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "neither sha256sum nor shasum is installed" >&2
        return 1
    fi
}

verify_sha256() {
    local path="$1"
    local expected="$2"
    local actual
    actual="$(sha256_file "${path}")"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "dataset hash mismatch for ${path}" >&2
        echo "expected ${expected}" >&2
        echo "actual   ${actual}" >&2
        exit 1
    fi
}

verify_sha256 "${TRAIN_PATH}" "${EXPECTED_TRAIN_SHA256}"
verify_sha256 "${VALIDATION_PATH}" "${EXPECTED_VALIDATION_SHA256}"
verify_sha256 "${CONFIG_PATH}" "${EXPECTED_CONFIG_SHA256}"

if [[ -f "${MANIFEST_PATH}" ]]; then
    echo "dataset manifest: ${MANIFEST_PATH}"
else
    echo "warning: ${MANIFEST_PATH} is absent; Parquet hashes still match" >&2
fi

RLLM_BIN="${RLLM_BIN:-${REPO_ROOT}/.venv/bin/rllm}"
if [[ ! -x "${RLLM_BIN}" ]]; then
    RLLM_BIN="$(command -v rllm || true)"
fi
if [[ -z "${RLLM_BIN}" || ! -x "${RLLM_BIN}" ]]; then
    echo "rllm executable not found; set RLLM_BIN or install the repository environment" >&2
    exit 1
fi

RLLM_PYTHON="${RLLM_PYTHON:-$(dirname -- "${RLLM_BIN}")/python}"
if [[ ! -x "${RLLM_PYTHON}" ]]; then
    echo "Python next to rllm was not found; set RLLM_PYTHON explicitly" >&2
    exit 1
fi

if ! git -C "${REPO_ROOT}" merge-base --is-ancestor "${REQUIRED_RLLM_BASE_REVISION}" HEAD; then
    echo "rLLM HEAD does not contain the required SFT base ${REQUIRED_RLLM_BASE_REVISION}" >&2
    exit 1
fi

"${RLLM_PYTHON}" - <<'PY'
import importlib.metadata
import json
import sys

expected = {
    "rllm": "0.3.0rc0",
    "fireworks-ai": "1.2.1",
    "fireworks-training-cookbook": "0.1.0",
    "tinker": "0.22.7",
    "tinker-cookbook": "0.4.2",
    "transformers": "4.57.6",
    "datasets": "5.0.1",
    "pyarrow": "23.0.1",
    "tokenizers": "0.22.2",
}
problems = []
actual_python = ".".join(str(part) for part in sys.version_info[:3])
if actual_python != "3.12.3":
    problems.append(f"Python {actual_python} != 3.12.3")
for package, wanted in expected.items():
    try:
        actual = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        problems.append(f"{package} is not installed")
        continue
    if actual != wanted:
        problems.append(f"{package} {actual} != {wanted}")

dist = importlib.metadata.distribution("fireworks-training-cookbook")
direct_url_raw = dist.read_text("direct_url.json")
direct_url = json.loads(direct_url_raw) if direct_url_raw else {}
actual_commit = direct_url.get("vcs_info", {}).get("commit_id")
expected_commit = "6b1232504fdacd1149895acf63b388ae792cb062"
if actual_commit != expected_commit:
    problems.append(
        "fireworks-training-cookbook commit "
        f"{actual_commit!r} != {expected_commit}"
    )

if problems:
    raise SystemExit("runtime mismatch:\n- " + "\n- ".join(problems))
PY

export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/../.hf-tokenizer-cache}"
if [[ "${NEMOTRON_SFT_OFFLINE:-1}" == "1" ]]; then
    TOKENIZER_REF="${HF_HUB_CACHE}/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/refs/main"
    if [[ ! -f "${TOKENIZER_REF}" || "$(<"${TOKENIZER_REF}")" != "${TOKENIZER_REVISION}" ]]; then
        echo "the pinned Nemotron tokenizer revision is not cached under ${HF_HUB_CACHE}" >&2
        echo "populate the cache first or set NEMOTRON_SFT_OFFLINE=0 for a non-identical online resolution" >&2
        exit 1
    fi
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ "${NEMOTRON_SFT_DRY_RUN:-0}" != "1" && -z "${FIREWORKS_API_KEY:-}" ]]; then
    echo "FIREWORKS_API_KEY is required for Fireworks SFT" >&2
    exit 1
fi

RUNS_DIR="${NEMOTRON_SFT_RUNS_DIR:-${REPO_ROOT}/artifacts/nemotron-swe/approx-a/runs}"
mkdir -p "${RUNS_DIR}"
RUN_STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG_PATH="${NEMOTRON_SFT_LOG_PATH:-${RUNS_DIR}/fireworks-sft-${RUN_STAMP}.log}"

COMMAND=(
    "${RLLM_BIN}" sft
    --backend fireworks
    --train-file "${TRAIN_PATH}"
    --val-file "${VALIDATION_PATH}"
    --model accounts/fireworks/models/nemotron-nano-3-30b-a3b
    --lora-rank 32
    --lr 1.0e-4
    --batch-size 9
    --epochs 1
    --max-length 65536
    --tokenize-method cumulative
    --lr-schedule cosine
    --val-freq 100
    --save-freq 250
    --project rllm-nemotron3-swe-sft
    --experiment nemotron3-nano-30b-a3b-lora32-fireworks
    --logger console
    --output "${RUNS_DIR}"
    --no-ui
    --config "${CONFIG_PATH}"
)

printf 'launch command:'
printf ' %q' "${COMMAND[@]}"
printf '\nlog: %s\n' "${LOG_PATH}"

if [[ "${NEMOTRON_SFT_DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

"${COMMAND[@]}" 2>&1 | tee "${LOG_PATH}"

#!/usr/bin/env bash
# Create an isolated environment matching the measured Nemotron Fireworks run.
# This intentionally overlays older measured SDK versions after installing the
# current rLLM Fireworks extra; it does not modify the repository's main venv.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_PATH="${SCRIPT_DIR}/nemotron_fireworks_sft_runtime.txt"
ENV_DIR="${NEMOTRON_SFT_ENV_DIR:-${REPO_ROOT}/.venv-nemotron-fireworks-sft}"

if [[ $# -gt 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi
if [[ ! -f "${RUNTIME_PATH}" ]]; then
    echo "missing measured runtime overlay: ${RUNTIME_PATH}" >&2
    exit 1
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    if [[ -e "${ENV_DIR}" ]]; then
        echo "${ENV_DIR} exists but is not a usable virtual environment" >&2
        echo "move it aside or set NEMOTRON_SFT_ENV_DIR to a new path" >&2
        exit 1
    fi
    uv venv --python 3.12.3 "${ENV_DIR}"
fi

actual_python="$(${ENV_DIR}/bin/python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
if [[ "${actual_python}" != "3.12.3" ]]; then
    echo "${ENV_DIR} uses Python ${actual_python}; expected 3.12.3" >&2
    echo "move it aside or set NEMOTRON_SFT_ENV_DIR to a new path" >&2
    exit 1
fi

uv pip install --python "${ENV_DIR}/bin/python" -e "${REPO_ROOT}[fireworks]"
uv pip install --python "${ENV_DIR}/bin/python" --no-deps --requirement "${RUNTIME_PATH}"

printf 'environment ready: %s\n' "${ENV_DIR}"
printf 'activate with: source %q\n' "${ENV_DIR}/bin/activate"

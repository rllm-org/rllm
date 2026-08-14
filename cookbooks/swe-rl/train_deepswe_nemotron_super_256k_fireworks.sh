#!/usr/bin/env bash
# Verify and launch the 256K DeepSWE -> Nemotron 3 Super Fireworks recipe.
# Set DEEPSWE_SUPER_DRY_RUN=1 to perform all local checks without provisioning.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/deepswe_nemotron3_super_256k_fireworks.yaml"
DATASET_DIR="${1:-${DEEPSWE_SUPER_DATASET_DIR:-${REPO_ROOT}/artifacts/deepswe/nemotron-super-256k}}"
TRAIN_PATH="${DATASET_DIR}/train.parquet"
VALIDATION_PATH="${DATASET_DIR}/validation.parquet"
CANARY_PATH="${DATASET_DIR}/canary-longest-two.parquet"
MANIFEST_PATH="${DATASET_DIR}/manifest.json"
MODEL="accounts/fireworks/models/nemotron-3-super-120b-a12b-bf16"
TOKENIZER_REPO="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
TOKENIZER_REVISION="d51eab0d1f979ebc26b546e634a04f450d99158e"
RLLM_REVISION="38207dc67e80ff2b7e2de721141b5f8fcbf0347b"
SHAPE="accounts/fireworks/trainingShapes/nemotron-3-super-120b-a12b-bf16-262k-lora"
MAX_LENGTH=262144

if [[ $# -gt 1 ]]; then
    echo "usage: $0 [DATASET_DIR]" >&2
    exit 2
fi
for path in "${CONFIG_PATH}" "${TRAIN_PATH}" "${VALIDATION_PATH}" "${CANARY_PATH}" "${MANIFEST_PATH}"; do
    if [[ ! -f "${path}" ]]; then
        echo "missing required file: ${path}" >&2
        exit 1
    fi
done

RLLM_BIN="${RLLM_BIN:-${REPO_ROOT}/.venv/bin/rllm}"
if [[ ! -x "${RLLM_BIN}" ]]; then
    echo "rLLM executable not found at ${RLLM_BIN}; set RLLM_BIN explicitly" >&2
    exit 1
fi
RLLM_PYTHON="${RLLM_PYTHON:-$(dirname -- "${RLLM_BIN}")/python}"
HF_BIN="${HF_BIN:-$(dirname -- "${RLLM_BIN}")/hf}"
if [[ ! -x "${HF_BIN}" ]]; then
    echo "Hugging Face CLI not found at ${HF_BIN}; set HF_BIN explicitly" >&2
    exit 1
fi
if ! git -C "${REPO_ROOT}" merge-base --is-ancestor "${RLLM_REVISION}" HEAD; then
    echo "recipe requires rLLM ${RLLM_REVISION} or a descendant" >&2
    exit 1
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
actual_rllm_root="$("${RLLM_PYTHON}" -c 'import pathlib, rllm; print(pathlib.Path(rllm.__file__).resolve().parents[1])')"
if [[ "${actual_rllm_root}" != "${REPO_ROOT}" ]]; then
    echo "RLLM_PYTHON imports ${actual_rllm_root}, expected ${REPO_ROOT}" >&2
    exit 1
fi

"${RLLM_PYTHON}" - "${MANIFEST_PATH}" "${TRAIN_PATH}" "${VALIDATION_PATH}" "${CANARY_PATH}" <<'PY'
import hashlib
import importlib.metadata
import json
import pathlib
import sys

manifest_path, train_path, validation_path, canary_path = map(pathlib.Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
expected = {
    "schema_version": "rllm.deepswe-nemotron-super-sft.v1",
    "renderer": "prime:nemotron-3",
    "renderers_version": "0.1.9",
    "max_length": 262144,
}
actual = {
    "schema_version": manifest["schema_version"],
    "renderer": manifest["rendering"]["renderer"],
    "renderers_version": importlib.metadata.version("renderers"),
    "transformers": importlib.metadata.version("transformers"),
    "tokenizers": importlib.metadata.version("tokenizers"),
    "max_length": manifest["conversion"]["max_length"],
}
expected["transformers"] = manifest["runtime"]["transformers"]
expected["tokenizers"] = manifest["runtime"]["tokenizers"]
if actual != expected:
    raise SystemExit(f"dataset/runtime contract mismatch: {actual!r} != {expected!r}")
for split, path in (("train", train_path), ("validation", validation_path), ("canary", canary_path)):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest[split]["sha256"]:
        raise SystemExit(f"{split} SHA-256 mismatch: {digest}")
PY

TOKENIZER_CACHE="${DEEPSWE_SUPER_TOKENIZER_CACHE:-${REPO_ROOT}/artifacts/deepswe/tokenizer-cache}"
TOKENIZER_REPO_CACHE="${TOKENIZER_CACHE}/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
TOKENIZER_SNAPSHOT="${TOKENIZER_REPO_CACHE}/snapshots/${TOKENIZER_REVISION}"
TOKENIZER_FILES=(chat_template.jinja config.json special_tokens_map.json tokenizer.json tokenizer_config.json)
TOKENIZER_SHA256=(
    575fb74f54ed264df9047d0ecce3c98938aae953fb4f50356675706264cbb68a
    699f34f0fc645d29ebffa5767fb59e6ae6ec98e3a4605485eb9913256d0df7e6
    e9435fefd6d838fd9fcbbc44b97a8e3ff322be7f6dfb7e4fd2468586574bb52b
    623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7
    10f93eabcb9b1602fbb991d6308e787ce1df28ee9cd7a1c6d1e8c3f338b957bc
)

missing=0
for filename in "${TOKENIZER_FILES[@]}"; do
    [[ -f "${TOKENIZER_SNAPSHOT}/${filename}" ]] || missing=1
done
if [[ "${missing}" == 1 ]]; then
    "${HF_BIN}" download "${TOKENIZER_REPO}" "${TOKENIZER_FILES[@]}" \
        --revision "${TOKENIZER_REVISION}" \
        --cache-dir "${TOKENIZER_CACHE}"
fi
for index in "${!TOKENIZER_FILES[@]}"; do
    actual="$(sha256sum "${TOKENIZER_SNAPSHOT}/${TOKENIZER_FILES[index]}" | awk '{print $1}')"
    if [[ "${actual}" != "${TOKENIZER_SHA256[index]}" ]]; then
        echo "tokenizer hash mismatch: ${TOKENIZER_FILES[index]}" >&2
        exit 1
    fi
done
mkdir -p "${TOKENIZER_REPO_CACHE}/refs"
printf '%s' "${TOKENIZER_REVISION}" > "${TOKENIZER_REPO_CACHE}/refs/main"
export HF_HUB_CACHE="${TOKENIZER_CACHE}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

RESOLVED_SHAPE="not-resolved-in-dry-run"
if [[ "${DEEPSWE_SUPER_DRY_RUN:-0}" != 1 ]]; then
    if [[ -z "${FIREWORKS_API_KEY:-}" ]]; then
        echo "FIREWORKS_API_KEY is required" >&2
        exit 1
    fi
    IFS='|' read -r RESOLVED_SHAPE SHAPE_ACCELERATORS SHAPE_ACCELERATOR < <("${RLLM_PYTHON}" - "${SHAPE}" <<'PY'
import os
import sys
from fireworks.training.sdk import TrainerJobManager

profile = TrainerJobManager(api_key=os.environ["FIREWORKS_API_KEY"]).resolve_training_profile(sys.argv[1])
if profile.trainer_mode != "LORA_TRAINER":
    raise SystemExit(f"shape is not LoRA-capable: {profile.trainer_mode}")
if profile.max_supported_context_length < 262144:
    raise SystemExit(f"shape context is only {profile.max_supported_context_length}")
print(f"{profile.training_shape_version}|{profile.accelerator_count}|{profile.accelerator_type}")
PY
    )
    echo "shape: ${RESOLVED_SHAPE} (${SHAPE_ACCELERATORS}x ${SHAPE_ACCELERATOR})"
fi

RUNS_DIR="${DEEPSWE_SUPER_RUNS_DIR:-${REPO_ROOT}/artifacts/deepswe/nemotron-super-256k-runs}"
mkdir -p "${RUNS_DIR}"
TRAIN_FILE="${TRAIN_PATH}"
BATCH_SIZE="${DEEPSWE_SUPER_BATCH_SIZE:-2}"
EXPERIMENT="nemotron3-super-120b-a12b-lora32-256k"
VAL_FREQ=50
DATA_ARGS=(--train-file "${TRAIN_FILE}" --val-file "${VALIDATION_PATH}")
EXTRA_ARGS=()
if [[ "${DEEPSWE_SUPER_CANARY:-0}" == 1 ]]; then
    TRAIN_FILE="${CANARY_PATH}"
    BATCH_SIZE=2
    EXPERIMENT="${EXPERIMENT}-capacity-canary"
    VAL_FREQ=-1
    DATA_ARGS=(--train-file "${TRAIN_FILE}" --max-examples 2)
fi
if [[ "${BATCH_SIZE}" != 1 && "${BATCH_SIZE}" != 2 ]]; then
    echo "only batch sizes 1 and 2 are supported by this 256K recipe" >&2
    exit 1
fi
if [[ "${DEEPSWE_SUPER_WANDB:-0}" == 1 ]]; then
    EXTRA_ARGS+=(--logger wandb)
fi

COMMAND=(
    "${RLLM_BIN}" sft
    --backend fireworks
    "${DATA_ARGS[@]}"
    --model "${MODEL}"
    --renderer nemotron3
    --lora-rank 32
    --lr 1.0e-4
    --batch-size "${BATCH_SIZE}"
    --epochs 1
    --max-length "${MAX_LENGTH}"
    --tokenize-method cumulative
    --lr-schedule constant
    --val-freq "${VAL_FREQ}"
    --save-freq -1
    --project rllm-deepswe-sft
    --experiment "${EXPERIMENT}"
    --no-ui
    --output "${RUNS_DIR}"
    --config "${CONFIG_PATH}"
    "${EXTRA_ARGS[@]}"
)

printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
if [[ "${DEEPSWE_SUPER_DRY_RUN:-0}" == 1 ]]; then
    "${RLLM_PYTHON}" - "${CONFIG_PATH}" "${CANARY_PATH}" <<'PY'
from omegaconf import OmegaConf

from rllm.data import Dataset
from rllm.trainer.agent_sft_trainer import AgentSFTTrainer
from rllm.trainer.sft import SFTSpec
from rllm.trainer.sft.tinker_backend import build_sft_data

import sys

config_path, canary_path = sys.argv[1:]
train = Dataset.load_data(canary_path)
overrides = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
spec = SFTSpec(
    model="accounts/fireworks/models/nemotron-3-super-120b-a12b-bf16",
    train_dataset=train,
    lr=1e-4,
    epochs=1,
    batch_size=2,
    max_length=262144,
    tokenize_method="cumulative",
    lora_rank=32,
    save_freq=-1,
    val_freq=-1,
    project="rllm-deepswe-sft",
    experiment="local-preflight",
    output_dir="/tmp/deepswe-super-local-preflight",
    overrides=overrides,
)
backend = AgentSFTTrainer(spec, backend="fireworks").prepare()
_, dataset, _ = build_sft_data(backend.config, train, None)
dataset.preflight(label="canary", planned_batches=[(0, 0)])
batch = dataset.get_batch(0)
lengths = [len(datum.model_input.to_ints()) for datum in batch]
loss_tokens = [sum(float(weight) > 0 for weight in datum.loss_fn_inputs["weights"].data) for datum in batch]
weight_mass = sum(float(weight) for datum in batch for weight in datum.loss_fn_inputs["weights"].data)
if len(batch) != 2 or max(lengths) >= 262144 or not 0.999999 <= weight_mass <= 1.000001:
    raise SystemExit(f"local production preflight failed: {len(batch)=}, {lengths=}, {weight_mass=}")
print(f"local production preflight: lengths={lengths}, loss_tokens={loss_tokens}, weight_mass={weight_mass}")
PY
    echo "capacity gate: run DEEPSWE_SUPER_CANARY=1 before the paid batch-2 launch"
    exit 0
fi
CANARY_MARKER="${RUNS_DIR}/.super-256k-batch2-canary-passed"
RECIPE_REVISION="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
CANARY_CONTRACT="$("${RLLM_PYTHON}" - "${MANIFEST_PATH}" "${CONFIG_PATH}" "${RESOLVED_SHAPE}" "${RECIPE_REVISION}" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path, config_path = map(pathlib.Path, sys.argv[1:3])
manifest = json.loads(manifest_path.read_text())
contract = {
    "canary_sha256": manifest["canary"]["sha256"],
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "max_length": 262144,
    "model": "accounts/fireworks/models/nemotron-3-super-120b-a12b-bf16",
    "recipe_revision": sys.argv[4],
    "renderer": manifest["rendering"],
    "resolved_shape": sys.argv[3],
}
payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(payload).hexdigest())
PY
)"
if [[ "${DEEPSWE_SUPER_CANARY:-0}" != 1 && "${BATCH_SIZE}" == 2 ]] && \
    [[ ! -f "${CANARY_MARKER}" || "$(<"${CANARY_MARKER}")" != "${CANARY_CONTRACT}" ]]; then
    echo "batch 2 requires a matching two-longest-row capacity canary:" >&2
    echo "  DEEPSWE_SUPER_CANARY=1 $0 ${DATASET_DIR}" >&2
    exit 1
fi
LOG_PATH="${RUNS_DIR}/${EXPERIMENT}-$(date -u +%Y%m%d-%H%M%S).log"
"${COMMAND[@]}" 2>&1 | tee "${LOG_PATH}"
if [[ "${DEEPSWE_SUPER_CANARY:-0}" == 1 ]]; then
    marker_tmp="${CANARY_MARKER}.tmp.$$"
    printf '%s' "${CANARY_CONTRACT}" > "${marker_tmp}"
    mv "${marker_tmp}" "${CANARY_MARKER}"
fi

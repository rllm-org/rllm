#!/usr/bin/env bash
# Run a bounded Tinker LoRA pilot on the pinned 256K DeepSWE SFT dataset.
# Default is deliberately 25 optimizer steps; set DEEPSWE_SUPER_TINKER_FULL=1
# only after reviewing the pilot and its W&B curve.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/deepswe_nemotron3_super_256k_tinker.yaml"
DATASET_DIR="${1:-${DEEPSWE_SUPER_DATASET_DIR:-${REPO_ROOT}/../deepswe-sft-prepared/nemotron-super-256k-repaired}}"
TRAIN_PATH="${DATASET_DIR}/train.parquet"
VALIDATION_PATH="${DATASET_DIR}/validation.parquet"
CANARY_PATH="${DATASET_DIR}/canary-longest-two.parquet"
MANIFEST_PATH="${DATASET_DIR}/manifest.json"
MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144"
TOKENIZER_REPO="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
TOKENIZER_REVISION="d51eab0d1f979ebc26b546e634a04f450d99158e"
MAX_LENGTH=262144

if [[ $# -gt 1 ]]; then
    echo "usage: $0 [DATASET_DIR]" >&2
    exit 2
fi
for path in "${CONFIG_PATH}" "${TRAIN_PATH}" "${VALIDATION_PATH}" "${CANARY_PATH}" "${MANIFEST_PATH}"; do
    [[ -f "${path}" ]] || { echo "missing required file: ${path}" >&2; exit 1; }
done

RLLM_BIN="${RLLM_BIN:-${REPO_ROOT}/.venv/bin/rllm}"
[[ -x "${RLLM_BIN}" ]] || { echo "rLLM executable not found at ${RLLM_BIN}; set RLLM_BIN explicitly" >&2; exit 1; }
RLLM_PYTHON="${RLLM_PYTHON:-$(dirname -- "${RLLM_BIN}")/python}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
actual_rllm_root="$("${RLLM_PYTHON}" -c 'import pathlib, rllm; print(pathlib.Path(rllm.__file__).resolve().parents[1])')"
[[ "${actual_rllm_root}" == "${REPO_ROOT}" ]] || { echo "RLLM_PYTHON imports ${actual_rllm_root}, expected ${REPO_ROOT}" >&2; exit 1; }

"${RLLM_PYTHON}" - "${MANIFEST_PATH}" "${TRAIN_PATH}" "${VALIDATION_PATH}" "${CANARY_PATH}" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path, train_path, validation_path, canary_path = map(pathlib.Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
if manifest["conversion"]["max_length"] != 262144 or manifest["train"]["rows"] != 222:
    raise SystemExit("unexpected DeepSWE 256K manifest contract")
for split, path in (("train", train_path), ("validation", validation_path), ("canary", canary_path)):
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[split]["sha256"]:
        raise SystemExit(f"{split} SHA-256 mismatch")
PY

TOKENIZER_CACHE="${DEEPSWE_SUPER_TOKENIZER_CACHE:-${REPO_ROOT}/artifacts/deepswe/tokenizer-cache}"
TOKENIZER_DIR="${DEEPSWE_SUPER_TOKENIZER_DIR:-${TOKENIZER_CACHE}/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/${TOKENIZER_REVISION}}"
TOKENIZER_FILES=(chat_template.jinja config.json special_tokens_map.json tokenizer.json tokenizer_config.json)
for filename in "${TOKENIZER_FILES[@]}"; do
    [[ -f "${TOKENIZER_DIR}/${filename}" ]] || { echo "missing pinned tokenizer file: ${TOKENIZER_DIR}/${filename}" >&2; exit 1; }
done
export HF_HUB_CACHE="${TOKENIZER_CACHE}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

BATCH_SIZE="${DEEPSWE_SUPER_TINKER_BATCH_SIZE:-2}"
LR="${DEEPSWE_SUPER_TINKER_LR:-2.0e-4}"
EPOCHS="${DEEPSWE_SUPER_TINKER_EPOCHS:-10}"
WARMUP_STEPS="${DEEPSWE_SUPER_TINKER_WARMUP_STEPS:-10}"
VAL_FREQ="${DEEPSWE_SUPER_TINKER_VAL_FREQ:-25}"
SAVE_FREQ="${DEEPSWE_SUPER_TINKER_SAVE_FREQ:-25}"
PROJECT="${DEEPSWE_SUPER_TINKER_PROJECT:-rllm-deepswe-sft}"
WANDB_ENABLED="${DEEPSWE_SUPER_TINKER_WANDB:-1}"
if [[ "${DEEPSWE_SUPER_TINKER_FULL:-0}" == 1 ]]; then
    MAX_STEPS=""
else
    MAX_STEPS="${DEEPSWE_SUPER_TINKER_MAX_STEPS:-25}"
fi

case "${BATCH_SIZE}" in
    1|2|4|8) ;;
    *) echo "DEEPSWE_SUPER_TINKER_BATCH_SIZE must be one of: 1, 2, 4, 8" >&2; exit 2 ;;
esac
[[ "${WANDB_ENABLED}" == 0 || "${WANDB_ENABLED}" == 1 ]] || { echo "DEEPSWE_SUPER_TINKER_WANDB must be 0 or 1" >&2; exit 2; }
[[ -z "${MAX_STEPS}" || "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "DEEPSWE_SUPER_TINKER_MAX_STEPS must be a positive integer" >&2; exit 2; }

RECIPE_REVISION="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
DATASET_REVISION="$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')"
HORIZON_TAG="e${EPOCHS}${MAX_STEPS:+-s${MAX_STEPS}}"
EXPERIMENT="deepswe-n3super-tinker-b${BATCH_SIZE}-lr${LR}-${HORIZON_TAG}-constant-wu${WARMUP_STEPS}-d${DATASET_REVISION:0:8}-r${RECIPE_REVISION:0:8}"
EXPERIMENT="${EXPERIMENT//[^[:alnum:]._-]/-}"
RUNS_DIR="${DEEPSWE_SUPER_TINKER_RUNS_DIR:-${REPO_ROOT}/artifacts/deepswe/nemotron-super-256k-tinker-runs}"
OUTPUT_DIR="${DEEPSWE_SUPER_TINKER_OUTPUT_DIR:-${RUNS_DIR}/${EXPERIMENT}}"
mkdir -p "${RUNS_DIR}"
if [[ -e "${OUTPUT_DIR}" && "${DEEPSWE_SUPER_TINKER_RESUME:-0}" != 1 ]]; then
    echo "output already exists; set DEEPSWE_SUPER_TINKER_RESUME=1 only to resume its matching checkpoint, or choose a new output" >&2
    exit 1
fi

RUN_CONFIG_PATH="$(mktemp "${TMPDIR:-/tmp}/deepswe-super-tinker.XXXXXX.yaml")"
trap 'rm -f -- "${RUN_CONFIG_PATH}"' EXIT
"${RLLM_PYTHON}" - "${CONFIG_PATH}" "${RUN_CONFIG_PATH}" "${MAX_STEPS}" "${WARMUP_STEPS}" "${TOKENIZER_DIR}" <<'PY'
import sys
from omegaconf import OmegaConf

source, destination, max_steps, warmup_steps, tokenizer_dir = sys.argv[1:]
override = {
    "model": {"tokenizer_model": tokenizer_dir},
    "optim": {"warmup_steps": int(warmup_steps)},
    "trainer": {"max_steps": int(max_steps) if max_steps else None},
}
OmegaConf.save(OmegaConf.merge(OmegaConf.load(source), OmegaConf.create(override)), destination)
PY

EXTRA_ARGS=()
if [[ "${WANDB_ENABLED}" == 1 ]]; then
    EXTRA_ARGS+=(--logger wandb)
fi
COMMAND=(
    "${RLLM_BIN}" sft
    --backend tinker
    --train-file "${TRAIN_PATH}"
    --val-file "${VALIDATION_PATH}"
    --model "${MODEL}"
    --renderer nemotron3
    --lora-rank 32
    --lr "${LR}"
    --batch-size "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --max-length "${MAX_LENGTH}"
    --tokenize-method cumulative
    --lr-schedule constant
    --val-freq "${VAL_FREQ}"
    --save-freq "${SAVE_FREQ}"
    --project "${PROJECT}"
    --experiment "${EXPERIMENT}"
    --no-ui
    --output "${OUTPUT_DIR}"
    --config "${RUN_CONFIG_PATH}"
    "${EXTRA_ARGS[@]}"
)
printf 'command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
printf 'tinker model SKU: %s (LoRA, 256K)\n' "${MODEL}"
printf 'run config: output=%q max_steps=%s warmup_steps=%s\n' "${OUTPUT_DIR}" "${MAX_STEPS:-null}" "${WARMUP_STEPS}"
printf 'tracking: wandb=%s project=%q experiment=%q\n' "${WANDB_ENABLED}" "${PROJECT}" "${EXPERIMENT}"

if [[ "${DEEPSWE_SUPER_TINKER_DRY_RUN:-0}" == 1 ]]; then
    "${RLLM_PYTHON}" - "${RUN_CONFIG_PATH}" "${CANARY_PATH}" "${TRAIN_PATH}" "${BATCH_SIZE}" <<'PY'
import sys
from omegaconf import OmegaConf
from rllm.data import Dataset
from rllm.trainer.agent_sft_trainer import AgentSFTTrainer
from rllm.trainer.sft import SFTSpec
from rllm.trainer.sft.tinker_backend import build_sft_data

config_path, canary_path, train_path, batch_size = sys.argv[1:]
batch_size = int(batch_size)
dataset_path = canary_path if batch_size <= 2 else train_path
train = Dataset.load_data(dataset_path)
overrides = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
spec = SFTSpec(model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144", train_dataset=train, lr=2e-4, epochs=1, batch_size=batch_size, max_length=262144, tokenize_method="cumulative", lora_rank=32, save_freq=-1, val_freq=-1, project="rllm-deepswe-sft", experiment="tinker-local-preflight", output_dir="/tmp/deepswe-super-tinker-local-preflight", overrides=overrides)
backend = AgentSFTTrainer(spec, backend="tinker").prepare()
_, dataset, _ = build_sft_data(backend.config, train, None)
dataset.preflight(label="canary", planned_batches=[(0, 0)])
batch = dataset.get_batch(0)
lengths = [datum.model_input.length for datum in batch]
weight_mass = sum(float(weight) for datum in batch for weight in datum.loss_fn_inputs["weights"].data)
if len(batch) != batch_size or max(lengths) >= 262144 or not 0.999999 <= weight_mass <= 1.000001:
    raise SystemExit(f"Tinker local preflight failed: {len(batch)=}, {lengths=}, {weight_mass=}")
print(f"Tinker local preflight: lengths={lengths}, weight_mass={weight_mass}")
PY
    exit 0
fi

[[ -n "${TINKER_API_KEY:-}" ]] || { echo "TINKER_API_KEY is required to start training" >&2; exit 1; }
"${COMMAND[@]}"

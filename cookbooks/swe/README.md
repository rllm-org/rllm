# SWE Agent

Software-engineering agent flow for rLLM that runs **mini-swe-agent** inside
Modal sandboxes, generates patches, and grades them against SWE-style
benchmarks.

## Overview

This cookbook keeps the original SWE package structure while fitting the rLLM
cookbook layout. The agent uses native OpenAI-compatible tool calling for bash
commands, supports context compaction, records clean trajectories, and can run
unchanged for evaluation or veRL training through the rLLM gateway.

Supported datasets:

| Dataset | Eval type | Notes |
|---|---|---|
| `swe_bench_pro` | `swebench_pro` | Uses the SWE-bench Pro harness from the cookbook submodule |
| `swe_bench_multilingual` | `swebench` | Uses the installed `swebench` harness |
| `swe_smith*` | `swesmith` | Uses the installed `swesmith` package, removes tests during rollout, and grades in a fresh sandbox |
| `swe_rebench_v2*` | `swe_rebench_v2` | Reuses the rollout sandbox for grading; set `SWE_REBENCH_V2_PATH` to a local checkout for parser code |

## Architecture

```
SWEAgentFlow.run(task, config)
  |
  +-- create Modal sandbox from task["docker_image"]
  +-- run mini-swe-agent DefaultAgent with OpenAIClientModel
  |     |
  |     +-- chat.completions.create(..., tools=[bash])
  |     +-- execute bash action in sandbox
  |     +-- compact context when configured token limits are reached
  |
  +-- return Episode with patch, exit_status, messages, segments

SWEEvaluator.evaluate(task, episode)
  |
  +-- route by task["eval_type"]
  +-- run dataset-specific grader
  +-- return EvalOutput(reward, is_correct, pass/fail signals)
```

## Installation

From the rLLM repo root:

```bash
uv pip install -e ".[verl,swe]"
uv pip install -e cookbooks/swe
git submodule update --init --recursive cookbooks/swe/external/SWE-bench_Pro-os
```

SWE-bench Pro is intentionally the only submodule in this cookbook. SWE-smith
is installed as a package dependency. SWE-rebench V2 is optional; point
`SWE_REBENCH_V2_PATH` at a local checkout before evaluating those tasks.

Modal authentication is required for live rollouts and grading:

```bash
modal setup
```

Create a `.env` file in `cookbooks/swe/` or the process environment with the
API keys used by the model endpoint and Modal.

## Qwen3.5 veRL Environment

The 9B Megatron training launchers expect a veRL/vLLM environment with Qwen3.5
support. The default cookbook-local environment path is:

```bash
cookbooks/swe/.venv-verl-vllm018/.venv
```

Build or refresh it with the cookbook setup script:

```bash
bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
```

The script installs the pinned Qwen3.5 stack used by these launchers:
`torch==2.10.0+cu129`, `vllm==0.18.0`, `transformers==5.3.0`,
`megatron-core==0.17.0`, `megatron-bridge==0.4.0`,
`flash-attn==2.8.3`, TransformerEngine `release_v2.12`, Apex,
`flash-linear-attention==0.4.1`, NVIDIA ModelOpt, veRL, rLLM, the model
gateway, and this cookbook.

Useful overrides:

```bash
VENV_ROOT=/path/to/venv-root bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
VERL_PATH=/path/to/verl bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
MAX_JOBS=64 bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
FLASH_ATTN_CUDA_ARCHS=100 bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
NVTE_CUDA_ARCHS=100 bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
RUN_SMOKE_TEST=0 bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
```

The source builds require a CUDA toolkit with `nvcc` on `PATH`; on the current
H100/B200 nodes `/usr/local/cuda/bin` is the expected source. The script unsets
`PYTHONPATH` while building so host packages do not leak into the venv.
For B200-only builds, keep `FLASH_ATTN_CUDA_ARCHS=100` and
`NVTE_CUDA_ARCHS=100`; the package defaults also compile older architectures
and are much slower.

On some B200 images, `ldconfig` resolves `libcuda.so.1` from the CUDA compat
directory before the real driver library. That makes Torch report CUDA error
803 even though `nvidia-smi` works. The setup and training scripts now prefer
`/usr/lib/x86_64-linux-gnu/libcuda.so.1` when it exists.

## Data

Prepare evaluation JSON:

```bash
python -m swe.prepare_data \
    --dataset swe_bench_pro \
    --split test \
    --output data/swe_bench_pro.json
```

Register datasets for rLLM training:

```bash
python -m swe.prepare_rllm_data --dataset swe_smith --split train
python -m swe.prepare_rllm_data --dataset swe_bench_multilingual --split test
python -m swe.scripts.prepare_filtered_mix
```

`prepare_rllm_data.py` writes to rLLM's `DatasetRegistry`.

## Eval

```bash
python -m swe.scripts.run_eval \
    --dataset swe_bench_pro \
    --model anthropic/claude-sonnet-4-5-20250929 \
    --slice 0:5 \
    --output_dir results/smoke \
    --verbose
```

For OpenAI-compatible hosted vLLM:

```bash
export HOSTED_VLLM_API_BASE=http://localhost:8000/v1
export HOSTED_VLLM_API_KEY=fake-api-key

python -m swe.scripts.run_n_eval \
    --dataset swe_smith_py \
    --model hosted_vllm/Qwen/Qwen3.5-9B \
    --n_runs 3 \
    --n_parallel 20 \
    --output_dir results/swe_smith_py_pass_at_3
```

## Training

The maintained launchers are the 9B Megatron runs:

```bash
bash cookbooks/swe/swe/training_scripts/run_swe_training_9b_megatron.sh
bash cookbooks/swe/swe/training_scripts/run_swe_training_9b_megatron_h100.sh
```

Both scripts run `python -m swe.scripts.train_swe_verl` and accept Hydra
overrides as trailing arguments. They resolve paths relative to this cookbook
and expect a veRL-capable virtualenv to be active, available at `/tmp/verl_venv`,
available through `VENV_DIR`, available at the cookbook-local setup path, or
available under `verl/.venv` in the rLLM checkout.

## Tests

```bash
pytest cookbooks/swe/tests -q
bash -n cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
bash -n cookbooks/swe/swe/training_scripts/run_swe_training_9b_megatron.sh
bash -n cookbooks/swe/swe/training_scripts/run_swe_training_9b_megatron_h100.sh
```

The pytest suite is unit-style and avoids live Modal grading. Dataset grading
smoke tests should be run manually with small slices when Modal credentials are
available.

## Files

| Path | Description |
|---|---|
| `swe/agent_flow.py` | `SWEAgentFlow` and `ProgressLoggingAgent` rollout logic |
| `swe/openai_model.py` | OpenAI-compatible mini-swe-agent model wrapper |
| `swe/evaluator.py` | Dataset-specific grading router |
| `swe/prepare_data.py` | Converts supported datasets into rollout task JSON |
| `swe/prepare_rllm_data.py` | Registers supported datasets in rLLM `DatasetRegistry` |
| `swe/scripts/run_eval.py` | Single-pass evaluation entry point |
| `swe/scripts/run_n_eval.py` | Multi-sample pass@k evaluation entry point |
| `swe/scripts/train_swe_verl.py` | veRL training entry point |
| `swe/scripts/prepare_filtered_mix.py` | Builds the SWE-smith filtered training mix |
| `scripts/setup_verl_vllm018_qwen35.sh` | Builds the Qwen3.5 veRL/vLLM training venv |
| `swe/training_scripts/` | Production 9B Megatron launchers |
| `swe/tasks/` | SWE-bench Pro, SWE-bench Multilingual, SWE-smith, and SWE-rebench V2 graders |
| `swe/config/` | Agent, Modal, compaction, and veRL Hydra config |
| `tests/` | Unit tests for model wrapping, compaction, trajectories, and retry patching |

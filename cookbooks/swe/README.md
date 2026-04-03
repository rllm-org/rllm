# SWE Flow

A multi-turn coding agent for rLLM that trains on SWE-bench datasets using the **AgentFlow protocol**. Supports SWE-bench Pro, SWE-bench Multilingual, and SWE-smith.

## Overview

A multi-step agent that receives a bug report, creates a Modal cloud sandbox, runs mini-swe-agent v2 to generate a patch, and grades it against the dataset's test suite. Uses a plain `OpenAI` client with the mini-swe-agent protocol.

During training, `config.base_url` points to the model gateway which transparently captures token IDs and logprobs. During eval, it points directly to the model provider. The agent code is identical in both cases.

## Architecture

```
SWEAgentFlow.run(task, config)
  |
  +-- Create Modal sandbox from task's Docker image
  +-- Run mini-swe-agent v2 (multi-step bash loop)
  |     +-- OpenAIClientModel(base_url=config.base_url)
  |     +-- ProgressLoggingAgent (step logging, timeout, format error handling)
  +-- Extract patch from agent submission
  +-- Episode(trajectories=[solver], artifacts={patch, exit_status, env, messages})

SWEEvaluator.evaluate(task, episode)
  |
  +-- Route by eval_type:
  |     +-- swebench_pro  -> eval_with_modal (SWE-bench_Pro-os)
  |     +-- swesmith      -> SWE-smith harness (fresh sandbox)
  |     +-- swebench      -> SWE-bench harness (reuse agent sandbox)
  +-- EvalOutput(reward=0|1, signals=[f2p_passed, f2p_total, p2p_passed, p2p_total])
```

## Installation

```bash
# From the rllm repo root
uv pip install -e ".[tinker]"                # rllm + tinker backend
uv pip install -e cookbooks/swe        # this cookbook
git submodule update --init --recursive      # SWE-bench_Pro-os
modal setup                                  # configure Modal credentials
```

## Datasets

| Dataset | eval_type | Source |
|---------|-----------|--------|
| `swe_bench_pro` | `swebench_pro` | CSV from SWE-bench_Pro-os (or HuggingFace fallback) |
| `swe_bench_multilingual` | `swebench` | `swe-bench/SWE-Bench_Multilingual` on HuggingFace |
| `swe_smith_py` | `swesmith` | `JWei05/SWE-smith-py-39471-filtered-for-problem-statements` |
| `swe_smith_go` | `swesmith` | `JWei05/SWE-smith-go-1629-filtered-for-problem-statements` |

## Eval

```bash
# OpenAI
python cookbooks/swe/eval.py \
    --base-url https://api.openai.com/v1 \
    --model gpt-4.1-mini \
    --dataset swe_bench_pro --slice 0:5 \
    --output_dir results/test/ -v

# Self-hosted vLLM
python cookbooks/swe/eval.py \
    --base-url http://localhost:8000/v1 \
    --api-key EMPTY \
    --model Qwen/Qwen3.5-35B-A3B \
    --dataset swe_smith_py --n_parallel 50 \
    --output_dir results/qwen/
```

## Training

### Register datasets first

```bash
python cookbooks/swe/prepare_tinker_data.py --dataset swe_smith_py --split train
python cookbooks/swe/prepare_tinker_data.py --dataset swe_bench_multilingual --split test
```

### Option 1: Python API

```bash
python cookbooks/swe/train.py \
    train_dataset=swe_smith_py \
    val_dataset=swe_bench_multilingual \
    model.name=Qwen/Qwen3-8B \
    training.group_size=8
```

### Option 2: Shell script (with defaults)

```bash
bash cookbooks/swe/train.sh
```

## Files

| File | Description |
|------|-------------|
| `swe_agent_flow.py` | `SWEAgentFlow` -- AgentFlow implementation (multi-turn sandbox agent) |
| `evaluator.py` | `SWEEvaluator` -- routes to dataset-specific graders |
| `openai_model.py` | `OpenAIClientModel` -- bridges OpenAI SDK to mini-swe-agent protocol |
| `environment.py` | Bootstrap patches, load YAML config, create Modal sandboxes |
| `swebench_pro.yaml` | Agent prompts, observation template, Modal env config |
| `data.py` | Dataset loading and normalization |
| `prepare_tinker_data.py` | Register datasets in rllm DatasetRegistry for training |
| `eval.py` | Evaluation entry point (EvalRunner) |
| `train.py` | Training entry point (AgentTrainer + Hydra) |
| `train.sh` | Shell wrapper with default training overrides |
| `tinker_unified.yaml` | Hydra training config (GRPO, LoRA, parallelism) |
| `tasks/` | Dataset-specific graders (swebench_pro, multilingual, swesmith) |
| `patches/` | SWE-ReX monkey-patches for Modal compatibility |
| `SWE-bench_Pro-os/` | Git submodule for SWE-bench Pro grading harness |
| `pyproject.toml` | Plugin metadata and entry points |

## Prerequisites

1. **Modal setup**: `modal setup`
2. **API keys**: Set `OPENAI_API_KEY` in environment or `.env` file
3. **SWE-bench_Pro-os**: `git submodule update --init --recursive`
4. **Docker images**: Pre-built on DockerHub (`jefzda/sweap-images`). No local Docker needed.

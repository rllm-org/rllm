# RL Training with Prime Intellect's Environment Hub

This example shows how to train RL agents using environments from Prime Intellect's [Environment Hub](https://app.primeintellect.ai/dashboard/environments) with the [verifiers](https://docs.primeintellect.ai/verifiers/overview) library.

## Overview

With this example you will:

1. Install and configure the **verifiers** library for RL environments
2. Download environments from Prime Intellect's Hub using the **Prime CLI**
3. Train agents using either the **verl** (local GPU) or **Tinker** (remote GPU) backend
4. Understand how rLLM wraps verifiers environments via `VerifiersWorkflow`

Under the hood, the integration works as:

- **VerifiersWorkflow**: Wraps rLLM's `RolloutEngine` as an `AsyncOpenAI`-compatible client
- **RolloutEngineAsyncClient**: Adapter that lets verifiers environments call rLLM for inference
- **Rubric scoring**: Verifiers environments define rubrics that score model outputs

---

## Setup

### 1. Install dependencies

Install rLLM with the verifiers extra:

```bash
uv sync --extra verifiers
```

This installs the `verifiers` library alongside rLLM.

For Tinker backend (remote GPU training), also install:

```bash
uv sync --extra tinker
```

### 2. Install Prime CLI

The Prime CLI lets you download environments from the Hub. Install it via:

```bash
uv tool install prime
```

Or with pipx:

```bash
pipx install prime
```

### 3. Login to Prime Intellect (optional)

Login is required for private environments:

```bash
prime login
```

This opens a browser for authentication. Public environments work without login.

### 4. Install an environment

Download an environment locally:

```bash
prime env install primeintellect/alphabet-sort
```

Format: `prime env install <owner>/<environment-name>`

For a specific version:

```bash
prime env install primeintellect/alphabet-sort@0.1.0
```

To see available installation methods:

```bash
prime env info primeintellect/alphabet-sort
```

See the [Prime Intellect docs](https://docs.primeintellect.ai/tutorials-environments/install) for more details.

---

## Environment Arguments

Each verifiers environment can accept custom arguments via `+verifiers.env_args`. These are passed directly to `vf.load_environment()`.

### Passing environment args

Use Hydra's nested syntax:

```bash
python -m examples.verifiers_env.train \
    +verifiers.env_id="primeintellect/alphabet-sort" \
    '+verifiers.env_args={max_turns: 10, difficulty: "hard"}' \
    ...
```

Or in the shell script:

```bash
VF_ENV_ARGS='max_turns: 20'

python -m examples.verifiers_env.train \
    +verifiers.env_id="${VF_ENV_ID}" \
    "+verifiers.env_args={${VF_ENV_ARGS}}" \
    ...
```

### Common environment arguments

| Argument        | Description                | Example                         |
| --------------- | -------------------------- | ------------------------------- |
| `max_turns`     | Maximum conversation turns | `10`                            |
| `tools`         | List of tools to enable    | `["search", "calculator"]`      |
| `system_prompt` | Custom system prompt       | `"You are a helpful assistant"` |

Check each environment's documentation for its specific arguments. You can also inspect an environment locally:

```python
import verifiers as vf

# See what arguments load_environment accepts
env = vf.load_environment("primeintellect/alphabet-sort")
print(env)  # Shows environment configuration
```

---

## Training with verl Backend (Local GPU)

The verl backend runs inference and training on your local GPUs using vLLM.

### Run training

```bash
cd examples/verifiers_env
bash train_verifiers.sh
```

Shell configuration:

```bash title="examples/verifiers_env/train_verifiers.sh"
--8<-- "examples/verifiers_env/train_verifiers.sh"
```

Key parameters to customize:

- `MODEL_PATH`: HuggingFace model path (e.g., `Qwen/Qwen3-4B`)
- `VF_ENV_ID`: Verifiers environment ID (e.g., `primeintellect/alphabet-sort`)
- `N_GPUS`: Number of GPUs to use
- `ROLLOUT_N`: Samples per prompt (GRPO group size)

---

## Training with Tinker Backend (Remote GPU)

The Tinker backend offloads inference and LoRA training to Prime Intellect's GPU service.

### Configure API key

Set your Tinker API key:

```bash
export TINKER_API_KEY=your_api_key_here
```

Or create a `.env` file in the example directory:

```bash
# examples/verifiers_env/.env
TINKER_API_KEY=your_api_key_here
```

### Run training

```bash
cd examples/verifiers_env
bash train_verifiers_tinker.sh
```

Shell configuration:

```bash title="examples/verifiers_env/train_verifiers_tinker.sh"
--8<-- "examples/verifiers_env/train_verifiers_tinker.sh"
```

Key Tinker-specific parameters:

- `model.name`: Model to fine-tune (e.g., `Qwen/Qwen3-4B-Instruct-2507`)
- `model.lora_rank`: LoRA rank for training
- `training.group_size`: GRPO group size
- `+backend=tinker`: Selects the Tinker backend

---

## Quick Test

To verify everything works with minimal resources:

```bash
cd examples/verifiers_env
bash test_train.sh
```

This runs a minimal training loop with:

- 20 samples (`+verifiers.max_samples=20`)
- Batch size of 4
- Single epoch

Test script:

```bash title="examples/verifiers_env/test_train.sh"
--8<-- "examples/verifiers_env/test_train.sh"
```

---

## Architecture

### How it works

```
┌─────────────────────────────────────────────────────────────┐
│                      AgentTrainer                           │
│  (orchestrates training loop, handles batching)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    VerifiersWorkflow                        │
│  - Receives task from trainer                               │
│  - Wraps RolloutEngine as AsyncOpenAI client                │
│  - Calls verifiers env.rollout()                            │
│  - Converts verifiers State → rllm Episode                  │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
┌─────────────────────────┐          ┌───────────────────────┐
│ RolloutEngineAsyncClient│          │   Verifiers Env       │
│ (AsyncOpenAI adapter)   │◄────────►│   (alphabet-sort,     │
│                         │          │    math, etc.)        │
└─────────────────────────┘          └───────────────────────┘
            │
            ▼
┌───────────────────────┐
│    RolloutEngine      │
│  (verl / Tinker)      │
└───────────────────────┘
```

### Key files

| File                        | Description                                                            |
| --------------------------- | ---------------------------------------------------------------------- |
| `train.py`                  | Main entry point, loads environment and starts trainer                 |
| `workflow.py`               | `VerifiersWorkflow` - bridges rLLM and verifiers                       |
| `openai_wrapper.py`         | `RolloutEngineAsyncClient` - makes RolloutEngine look like AsyncOpenAI |
| `train_verifiers.sh`        | Shell script for verl backend                                          |
| `train_verifiers_tinker.sh` | Shell script for Tinker backend                                        |

---

## Code Reference

### Training entry point

```python title="examples/verifiers_env/train.py"
--8<-- "examples/verifiers_env/train.py"
```

### VerifiersWorkflow

The workflow that bridges rLLM and verifiers:

```python title="examples/verifiers_env/workflow.py"
--8<-- "examples/verifiers_env/workflow.py"
```

### RolloutEngine AsyncOpenAI Wrapper

Adapter that makes RolloutEngine compatible with verifiers:

```python title="examples/verifiers_env/openai_wrapper.py"
--8<-- "examples/verifiers_env/openai_wrapper.py"
```

---

## Configuration Reference

### Verifiers-specific config

| Parameter                  | Description                         | Example                        |
| -------------------------- | ----------------------------------- | ------------------------------ |
| `+verifiers.env_id`        | Environment ID from Hub             | `primeintellect/alphabet-sort` |
| `+verifiers.env_args`      | Environment constructor args (JSON) | `'{"difficulty": "hard"}'`     |
| `+verifiers.sampling_args` | Sampling args for rollouts          | `'{"temperature": 0.7}'`       |
| `+verifiers.max_samples`   | Limit dataset size (for testing)    | `100`                          |
| `+backend`                 | Training backend                    | `verl` or `tinker`             |

### Common training config

| Parameter                  | Description         | Default               |
| -------------------------- | ------------------- | --------------------- |
| `data.train_batch_size`    | Prompts per batch   | `64`                  |
| `data.max_prompt_length`   | Max prompt tokens   | `2048`                |
| `data.max_response_length` | Max response tokens | `2048`                |
| `trainer.total_epochs`     | Training epochs     | `100`                 |
| `trainer.logger`           | Logging backends    | `['console','wandb']` |

### verl-specific config

| Parameter                      | Description            |
| ------------------------------ | ---------------------- |
| `actor_rollout_ref.model.path` | HuggingFace model path |
| `actor_rollout_ref.rollout.n`  | Samples per prompt     |
| `trainer.n_gpus_per_node`      | GPUs to use            |

### Tinker-specific config

| Parameter                | Description        |
| ------------------------ | ------------------ |
| `model.name`             | Model to fine-tune |
| `model.lora_rank`        | LoRA rank          |
| `training.group_size`    | GRPO group size    |
| `training.learning_rate` | Learning rate      |

---

## Troubleshooting

### Environment not found

If `prime env install` fails:

1. Check you're logged in: `prime login`
2. Verify the environment exists on the [Hub](https://app.primeintellect.ai/dashboard/environments)
3. Check the exact name format: `owner/environment-name`

### Tinker API key errors

Ensure the key is set:

```bash
echo $TINKER_API_KEY
```

Or check your `.env` file is in the right location.

### Dataset too large

Use `+verifiers.max_samples` to limit dataset size:

```bash
python -m examples.verifiers_env.train \
    +verifiers.env_id="primeintellect/alphabet-sort" \
    +verifiers.max_samples=1000 \
    ...
```

### Hydra config errors

When adding new config keys, use `+key=value` (with plus):

```bash
# Correct - adds new key
+backend=tinker
+verifiers.env_id="math"

# Wrong - tries to override existing key
backend=tinker  # Error: key not in struct
```

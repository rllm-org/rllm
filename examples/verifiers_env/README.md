# Verifiers Environment Training

Train RL agents using [Verifiers](https://github.com/verifiers-for-code/verifiers) environments with rLLM. This integration allows you to use any Verifiers-compatible environment for reinforcement learning training.

## Overview

This example provides:
- A workflow wrapper (`VerifiersWorkflow`) that bridges rLLM's training infrastructure with Verifiers environments
- Support for both **verl** (local GPU) and **Tinker** (remote inference) backends
- Seamless integration with Verifiers' rubric-based evaluation system

## Prerequisites

### 1. Install Verifiers

```bash
uv sync --extra verifiers
```

### 2. Install Prime CLI (for private environments)

```bash
uv tool install prime
```

### 3. Login to Prime (for private environments)

```bash
prime login
```

### 4. Install Environment

Install the specific Verifiers environment you want to use:

```bash
prime env install <env_id>
```

For example:
```bash
prime env install primeintellect/alphabet-sort
```

## Configuration

### Environment Arguments

Each Verifiers environment has its own specific arguments. Pass them via the config:

```bash
+verifiers.env_id="primeintellect/alphabet-sort" \
+verifiers.env_args.max_turns=5 \
+verifiers.env_args.difficulty="hard"
```

Check the specific environment's documentation for available arguments.

### Dataset Limiting

For testing or debugging, limit the dataset size:

```bash
+verifiers.max_samples=100
```

## Training Backends

### Option 1: Tinker Backend (Recommended for remote inference)

Tinker handles model inference remotely, supporting LoRA-based training.

**Setup:**

1. Get your Tinker API key and add to `.env`:
```bash
echo 'TINKER_API_KEY="your-api-key"' > examples/verifiers_env/.env
```

2. Run training:
```bash
bash examples/verifiers_env/train_verifiers_tinker.sh
```

**Key Tinker parameters:**
- `tinker_base_url`: Tinker service URL (null for default cloud)
- `model.name`: HuggingFace model path
- `model.lora_rank`: LoRA rank (default: 32)
- `training.group_size`: Rollouts per prompt for GRPO (default: 16)

### Option 2: verl Backend (Local GPU training)

verl manages vLLM internally for on-device training and inference.

**Requirements:**
- Local GPU(s) with sufficient VRAM
- verl installed (`uv pip install -e .[verl]`)

**Run training:**
```bash
bash examples/verifiers_env/train_verifiers.sh
```

**Key verl parameters:**
- `actor_rollout_ref.model.path`: Model to train
- `trainer.n_gpus_per_node`: Number of GPUs
- `actor_rollout_ref.rollout.gpu_memory_utilization`: GPU memory usage (0.0-1.0)

## Quick Test

Run a minimal training loop to verify everything works:

```bash
bash examples/verifiers_env/test_train.sh
```

This runs with:
- 20 samples only
- Batch size 4
- 1 epoch
- WandB logging enabled

## Training Scripts

| Script | Backend | Description |
|--------|---------|-------------|
| `train_verifiers.sh` | verl | Full training with local GPUs |
| `train_verifiers_tinker.sh` | Tinker | Full training with remote inference |
| `test_train.sh` | Tinker | Quick test run (20 samples, 1 epoch) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AgentTrainer                           │
│  (verl or Tinker backend)                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   VerifiersWorkflow                         │
│  - Wraps RolloutEngine as AsyncOpenAI client                │
│  - Converts Verifiers State → rLLM Episode                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               Verifiers Environment                         │
│  - Manages rollouts and multi-turn interactions             │
│  - Applies rubric-based scoring                             │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Reference

### Verifiers Config

| Parameter | Description | Default |
|-----------|-------------|---------|
| `verifiers.env_id` | Environment identifier | Required |
| `verifiers.env_args` | Environment-specific arguments | `{}` |
| `verifiers.sampling_args` | Sampling parameters for rollouts | `{}` |
| `verifiers.max_samples` | Limit dataset size (for testing) | None |

### Training Config

| Parameter | Description | Default |
|-----------|-------------|---------|
| `backend` | Training backend (`verl` or `tinker`) | `verl` |
| `data.train_batch_size` | Samples per batch | 64 |
| `trainer.total_epochs` | Number of training epochs | 10 |
| `trainer.logger` | Logging backends | `[console]` |

### Tinker-specific Config

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.name` | HuggingFace model path | Required |
| `model.lora_rank` | LoRA rank | 32 |
| `training.group_size` | Rollouts per prompt | 16 |
| `sampling.temperature` | Must be 1.0 for Tinker | 1.0 |

## Troubleshooting

### "Could not deserialize ATN" error
Reinstall omegaconf to fix antlr4 version mismatch:
```bash
uv pip install --reinstall omegaconf
```

### "huggingface-hub version" error
Upgrade transformers:
```bash
uv pip install --upgrade transformers
```

### Training runs forever
The dataset may be very large. Use `+verifiers.max_samples=100` to limit for testing.

### Model name not found
Ensure your model path is correct and accessible. For Tinker, the model must be available on the Tinker service.

## Files

```
examples/verifiers_env/
├── README.md                    # This file
├── train.py                     # Main training script
├── workflow.py                  # VerifiersWorkflow implementation
├── openai_wrapper.py            # RolloutEngine → AsyncOpenAI adapter
├── train_verifiers.sh           # verl backend training script
├── train_verifiers_tinker.sh    # Tinker backend training script
├── test_train.sh                # Quick test script
└── .env                         # API keys (not committed)
```

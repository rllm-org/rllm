# On-Policy Distillation for Math Reasoning

This example demonstrates **On-Policy Distillation (OPD)** using rLLM's unified trainer. OPD combines on-policy sampling with dense per-token feedback from a teacher model, achieving efficient post-training at a fraction of RL's cost.

## Overview

On-policy distillation works by:
1. Sampling trajectories from the **student** model (on-policy)
2. Grading each token using the **teacher** model's log probabilities (dense feedback)

This approach avoids the exposure bias of supervised fine-tuning while providing richer feedback than sparse RL rewards.

## Files

| File | Description |
|------|-------------|
| `prepare_deepmath_data.py` | Loads DeepMath-103K (train) and AIME 2024 (eval) datasets |
| `train_deepmath_distill_tinker.py` | Main training script using unified trainer with Tinker backend |
| `train_deepmath_distill_tinker.sh` | Launch script with recommended hyperparameters |

## Quick Start

### 1. Prepare the dataset

```bash
python -m examples.math_distill.prepare_deepmath_data
```

This downloads and registers:
- **Training**: [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) (~103K math problems)
- **Evaluation**: [AIME 2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) (30 competition problems)

### 2. Run training

```bash
bash examples/math_distill/train_deepmath_distill_tinker.sh
```

Or run directly with custom configs:

```bash
python -m examples.math_distill.train_deepmath_distill_tinker \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-8B-Base \
    model.lora_rank=128 \
    training.group_size=4 \
    training.learning_rate=1e-4 \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    rllm.algorithm.use_precomputed_advantage=true \
    rllm.algorithm.loss_fn=importance_sampling \
    rllm.trainer.logger=['console','wandb'] \
    rllm.trainer.project_name='opd-deepmath-8b-32b'
```

## Configuration

### Key parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.name` | Student model to train | `Qwen/Qwen3-8B-Base` |
| `model.lora_rank` | LoRA rank for efficient fine-tuning | `128` |
| `rllm.algorithm.use_precomputed_advantage` | Use workflow-computed advantages (required for OPD) | `true` |
| `rllm.algorithm.loss_fn` | Loss function for policy update | `importance_sampling` |
| `training.group_size` | Number of responses per prompt (train) | `4` |
| `validation.group_size` | Number of responses per prompt (val) | `8` |

### Teacher model

The teacher model is configured in `train_deepmath_distill_tinker.py`:

```python
teacher_model = "Qwen/Qwen3-32B"
```

The workflow uses `shared_tokenizer=True` since both Qwen3-8B and Qwen3-32B share the same tokenizer.

### Advantage clipping

Per-token advantages are clipped to prevent instability:

```python
workflow_args={
    "clip_min": -5.0,
    "clip_max": 5.0,
}
```

## How it works

The `DistillationWorkflow` computes per-token advantages as:

```
advantage[t] = log P_teacher(token_t) - log P_student(token_t)
```

Tokens where the student diverges from the teacher receive higher advantages, guiding the student to match the teacher's distribution on its own trajectories.

Because `rllm.algorithm.use_precomputed_advantage=true`, the unified trainer uses these pre-computed advantages directly instead of computing RL-style advantages.

## Requirements

- Tinker API access (for both student training and teacher inference)
- ~16GB GPU memory for 8B student model with LoRA

## References

- [On-Policy Distillation - Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [DeepMath-103K Dataset](https://huggingface.co/datasets/zwhe99/DeepMath-103K)
- [AIME 2024 Dataset](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)

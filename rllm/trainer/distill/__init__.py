"""Shared distillation utilities for cross-tokenizer teacher-student alignment."""

from rllm.trainer.distill.advantage import (
    compute_distill_reverse_kl,
    compute_step_distill_advantage,
    discounted_future_sum,
)
from rllm.trainer.distill.alignment import align_teacher_logprobs, visualize_alignment

__all__ = [
    "align_teacher_logprobs",
    "compute_distill_reverse_kl",
    "compute_step_distill_advantage",
    "discounted_future_sum",
    "visualize_alignment",
]

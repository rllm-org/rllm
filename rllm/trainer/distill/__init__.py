"""Shared distillation utilities for cross-tokenizer teacher-student alignment."""

from rllm.trainer.distill.advantage import compute_step_distill_advantage
from rllm.trainer.distill.alignment import align_teacher_logprobs, visualize_alignment

__all__ = [
    "align_teacher_logprobs",
    "compute_step_distill_advantage",
    "visualize_alignment",
]

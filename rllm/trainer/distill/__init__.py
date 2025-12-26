"""Shared distillation utilities for cross-tokenizer teacher-student alignment."""

from rllm.trainer.distill.alignment import (
    align_teacher_logprobs,
    build_byte_offsets,
    find_content_byte_ranges,
    visualize_alignment,
)

__all__ = [
    "align_teacher_logprobs",
    "build_byte_offsets",
    "find_content_byte_ranges",
    "visualize_alignment",
]

"""
Common utilities for rLLM trainers.

This module provides shared functionality across different trainer backends (verl, tinker, etc.).
"""

from rllm.trainer.common.rejection_sampling import (
    RejectionSamplingConfig,
    RejectionSamplingMetrics,
    RejectionSamplingState,
    apply_rejection_sampling_and_filter_groups,
)
from rllm.trainer.common.transform import (
    TransformConfig,
    build_trajectory_groups,
    impute_trajectory_names,
    transform_episodes_to_trajectory_groups,
    validate_and_propagate_rewards,
)

__all__ = [
    # Transform pipeline
    "transform_episodes_to_trajectory_groups",
    "TransformConfig",
    "build_trajectory_groups",
    "impute_trajectory_names",
    "validate_and_propagate_rewards",
    # Rejection sampling
    "RejectionSamplingConfig",
    "RejectionSamplingMetrics",
    "RejectionSamplingState",
    "apply_rejection_sampling_and_filter_groups",
]

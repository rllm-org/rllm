"""
Common utilities for rLLM trainers.

This module provides shared functionality across different trainer backends (verl, tinker, etc.).
"""

from rllm.experimental.common.advantage import compute_advantage_from_trajectory_groups
from rllm.experimental.common.config import (
    AlgorithmConfig,
    CompactFilteringConfig,
    RejectionSamplingConfig,
    TransformConfig,
    rLLMAdvantageEstimator,
)
from rllm.experimental.common.metrics import reduce_reward_metrics_by_trajectory_name
from rllm.experimental.common.rejection_sampling import (
    RejectionSamplingMetrics,
    RejectionSamplingState,
    apply_rejection_sampling_and_filtering,
)
from rllm.experimental.common.transform import (
    build_trajectory_groups,
    impute_trajectory_names,
    transform_episodes_to_trajectory_groups,
    validate_and_propagate_rewards,
)

__all__ = [
    # Config
    "CompactFilteringConfig",
    "RejectionSamplingConfig",
    "TransformConfig",
    "AlgorithmConfig",
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
    "apply_rejection_sampling_and_filtering",
    # Advantage computation
    "rLLMAdvantageEstimator",
    "compute_advantage_from_trajectory_groups",
    # Metrics
    "reduce_reward_metrics_by_trajectory_name",
]

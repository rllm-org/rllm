"""
Common utilities for rLLM trainers.

This module provides shared functionality across different trainer backends (verl, tinker, etc.).
"""

from rllm.trainer.algorithms.advantage import collect_reward_and_advantage_from_trajectory_groups
from rllm.trainer.algorithms.config import (
    AlgorithmConfig,
    AsyncTrainingConfig,
    CompactFilteringConfig,
    RejectionSamplingConfig,
    RolloutCorrectionConfig,
    TransformConfig,
    rLLMAdvantageEstimator,
)
from rllm.trainer.algorithms.metrics import reduce_metrics_by_trajectory_name, reduce_metrics_lists
from rllm.trainer.algorithms.performance import marked_timer, simple_timer
from rllm.trainer.algorithms.rejection_sampling import (
    RejectionSamplingMetrics,
    RejectionSamplingState,
    apply_rejection_sampling_and_filtering,
)
from rllm.trainer.algorithms.transform import transform_episodes_to_trajectory_groups
from rllm.trainer.algorithms.visualization import VisualizationConfig, colorful_print, colorful_warning, visualize_trajectory_last_steps

__all__ = [
    # Config
    "AsyncTrainingConfig",
    "CompactFilteringConfig",
    "RejectionSamplingConfig",
    "RolloutCorrectionConfig",
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
    "collect_reward_and_advantage_from_trajectory_groups",
    # Metrics
    "reduce_metrics_by_trajectory_name",
    "reduce_metrics_lists",
    # Performance
    "simple_timer",
    "marked_timer",
    # Visualization
    "VisualizationConfig",
    "colorful_print",
    "colorful_warning",
    "visualize_trajectory_last_steps",
]

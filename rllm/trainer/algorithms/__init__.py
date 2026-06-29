"""
Common utilities for rLLM trainers.

This module provides shared functionality across different trainer backends (verl, tinker, etc.).
"""

from rllm.trainer.algorithms.advantage import collect_reward_and_advantage_from_trajectory_groups
from rllm.trainer.algorithms.aux_loss import (  # deprecated shim — see rllm.trainer.algorithms.loss
    AUX_LOSS_REGISTRY,
    MASK_ACTION,
    MASK_OBSERVATION,
    AuxiliaryLoss,
    build_aux_losses,
    get_aux_loss,
    register_aux_loss,
)
from rllm.trainer.algorithms.config import (
    AlgorithmConfig,
    AsyncTrainingConfig,
    CompactFilteringConfig,
    RejectionSamplingConfig,
    RolloutCorrectionConfig,
    TransformConfig,
    rLLMAdvantageEstimator,
)
from rllm.trainer.algorithms.loss import (
    RLLM_LOSS_REGISTRY,
    LossContext,
    ResolvedTerm,
    get_loss,
    get_term_aux_mask,
    is_custom_loss,
    load_loss_plugins,
    register_loss,
    resolve_additive_terms,
    resolve_loss_terms,
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
    # Rejection sampling
    "RejectionSamplingConfig",
    "RejectionSamplingMetrics",
    "RejectionSamplingState",
    "apply_rejection_sampling_and_filtering",
    # Advantage computation
    "rLLMAdvantageEstimator",
    "collect_reward_and_advantage_from_trajectory_groups",
    # Auxiliary losses
    "AuxiliaryLoss",
    "EnvPredictionLoss",
    "build_aux_losses",
    "register_aux_loss",
    "get_aux_loss",
    "AUX_LOSS_REGISTRY",
    "MASK_ACTION",
    "MASK_OBSERVATION",
    # Unified custom losses
    "register_loss",
    "get_loss",
    "get_term_aux_mask",
    "is_custom_loss",
    "load_loss_plugins",
    "resolve_loss_terms",
    "resolve_additive_terms",
    "LossContext",
    "ResolvedTerm",
    "RLLM_LOSS_REGISTRY",
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

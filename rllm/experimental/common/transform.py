"""
Episode to TrajectoryGroup transformation pipeline.

This module provides functions to transform raw Episodes into TrajectoryGroups
for advantage computation in multi-agent/multi-trajectory workflows.

The pipeline handles:
1. Trajectory name imputation (for unnamed trajectories)
2. Reward validation and propagation
3. TrajectoryGroup construction with configurable grouping strategies
"""

import logging
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from rllm.agents.agent import Episode, Trajectory, TrajectoryGroup
from rllm.experimental.common.config import CompactFilteringConfig, TransformConfig
from rllm.workflows.workflow import TerminationReason

logger = logging.getLogger(__name__)
LOG_N_WARNINGS = 1


def _impute_trajectory_names(
    episodes: list[Episode],
    config: TransformConfig,
) -> list[str]:
    """
    Impute missing trajectory names using position-based naming. This imputation is in-place.

    For each episode, if a trajectory lacks a proper name, rename it to
    '{prefix}_{position}' (e.g., 'agent_0', 'agent_1').

    Args:
        episodes: List of episodes to process
        config: Transform configuration

    Returns:
        List of warning messages
    """
    warnings = []
    for episode in episodes:
        # Track position-based names needed
        new_trajs = []
        for traj_idx, trajectory in enumerate(episode.trajectories):
            if not trajectory.name or trajectory.name == config.default_traj_name:  # is unnamed
                if config.impute_missing_names:
                    new_name = f"{config.default_traj_name}_{traj_idx}"
                    warnings.append(f"Episode {episode.id}: Trajectory at position {traj_idx} renamed to '{new_name}'")
                    trajectory.name = new_name
                elif config.drop_unnamed_traj:
                    warnings.append(f"Episode {episode.id}: Trajectory at position {traj_idx} has no name and will be dropped")
                    continue
            new_trajs.append(trajectory)
        episode.trajectories = new_trajs

    return warnings


def _validate_and_propagate_rewards(
    groups: list[TrajectoryGroup],
    config: TransformConfig,
) -> list[str]:
    """
    Validate reward consistency and propagate rewards as needed.

    For broadcast=True mode:
    - Ensure each trajectory has a trajectory-level reward
    - If missing, propagate from last step reward

    For broadcast=False mode:
    - Ensure each step has a valid reward
    - Validate consistent step counts within groups

    Args:
        groups: List of trajectory groups to process
        config: Transform configuration

    Returns:
        List of warning messages
    """
    warnings = []

    for group in groups:
        if config.broadcast:
            num_missing_rewards = sum(traj.reward is None for traj in group.trajectories)
            assert num_missing_rewards == 0 or num_missing_rewards == len(group.trajectories), "Trajectories in a group must either ALL NOT have a trajectory-level reward or ALL have a trajectory-level reward"
            if num_missing_rewards > 0:
                for traj in group.trajectories:
                    assert len(traj.steps) > 0, "Trajectory within a group must have at least one step"
                    traj.reward = traj.steps[-1].reward
                    warnings.append(f"Trajectory {traj.name} in group {group.group_id} has no trajectory-level reward, propagated from last step reward")
        else:  # broadcast=False
            step_counts = [len(traj.steps) for traj in group.trajectories]
            assert len(set(step_counts)) == 1, "Trajectories in a group must have the same number of steps when broadcast=False"

    return warnings


def _build_trajectory_groups(episodes: list[Episode], compact_filtering_config: CompactFilteringConfig | None = None) -> list[TrajectoryGroup]:
    """
    Build TrajectoryGroups from episodes based on the configured grouping strategy.

    Args:
        episodes: List of episodes to group

    Returns:
        List of TrajectoryGroups
    """
    trajectories_by_name: dict[str, list[Trajectory]] = defaultdict(list)
    metadata_by_name: dict[str, list[dict]] = defaultdict(list)

    for episode in episodes:
        termination_reason = episode.termination_reason or TerminationReason.UNKNOWN
        # skip episode if it should be masked by compact filtering
        if compact_filtering_config and compact_filtering_config.should_mask(termination_reason):
            continue
        task_id = episode.task_id
        for trajectory in episode.trajectories:
            if len(trajectory.steps) == 0:
                continue
            trajectories_by_name[f"{task_id}:{trajectory.name}"].append(trajectory)
            metadata_by_name[f"{task_id}:{trajectory.name}"].append(
                {
                    "task_id": episode.task_id,
                    "rollout_idx": episode.rollout_idx,
                    "termination_reason": episode.termination_reason,
                    "is_correct": episode.is_correct,
                }
            )

    groups = []
    for name, trajectories in trajectories_by_name.items():
        groups.append(
            TrajectoryGroup(
                trajectories=trajectories,
                group_id=name,
                metadata=metadata_by_name[name],
            )
        )
    return groups


"""
Functionalities related to transforming episodes to trajectory groups. We will provide a default implementation with clear documentations
on the transformation logic. At the same time, we also allow user-defined "hooks" to override the default behaviors.
"""


def _get_transform_metrics(episodes: list[Episode], groups: list[TrajectoryGroup], prefix: str = "grouping") -> dict:
    """
    Get metrics for the transformation pipeline.
    """
    group_sizes_before = np.array([len(episode.trajectories) for episode in episodes])
    group_sizes = np.array([len(group.trajectories) for group in groups])
    metrics = dict()
    metrics[f"{prefix}/num_trajs_before_filter"] = group_sizes_before.sum()
    metrics[f"{prefix}/num_trajs_after_filter"] = group_sizes.sum()
    metrics[f"{prefix}/num_groups"] = len(groups)
    metrics[f"{prefix}/avg_group_size"] = group_sizes.mean()
    metrics[f"{prefix}/max_group_size"] = group_sizes.max()
    metrics[f"{prefix}/min_group_size"] = group_sizes.min()
    return metrics


def _default_trajectory_grouping_hook(episodes: list[Episode], transform_config: TransformConfig | None = None, compact_filtering_config: CompactFilteringConfig | None = None) -> list[TrajectoryGroup]:
    """
    Default trajectory grouping hook.

    A trajectory grouping hook needs to take care of two major things:
    1. Creation of a list of trajectory groups from a list of episodes
    2. Validation and propagation of rewards for the trajectory groups

    Note: the `Trajectory` objects, when passed from within the episodes to the trajectory groups,
    should not be copied (i.e. only reference is passed). This is explicitly enforced & checked in the
    `transform_episodes_to_trajectory_groups` function.
    """
    trajectory_groups = _build_trajectory_groups(episodes, compact_filtering_config)  # part 1
    reward_warnings = _validate_and_propagate_rewards(trajectory_groups, transform_config)  # part 2

    for warning in reward_warnings[:LOG_N_WARNINGS]:
        logger.warning(warning)

    if len(reward_warnings) > LOG_N_WARNINGS:
        logger.warning(f"Skipping {len(reward_warnings) - LOG_N_WARNINGS} more similar warnings with reward validation")

    return trajectory_groups


def transform_episodes_to_trajectory_groups(
    episodes: list[Episode],
    transform_config: TransformConfig | None = None,
    compact_filtering_config: CompactFilteringConfig | None = None,
    metrics_prefix: str = "grouping",
    trajectory_grouping_hook: Callable[[list[Episode], TransformConfig | None, CompactFilteringConfig | None], list[TrajectoryGroup]] = _default_trajectory_grouping_hook,
) -> tuple[list[TrajectoryGroup], dict]:
    """
    Transform a list of Episodes into a list of TrajectoryGroups for advantage computation.

    This is the main entry point for the transformation pipeline. It performs:
    1. Trajectory name imputation (for unnamed trajectories)
    2. Reward validation and propagation
    3. TrajectoryGroup construction

    Args:
        episodes: List of Episodes from workflow execution
        transform_config: Transform configuration (uses defaults if not provided)
        compact_filtering_config: Compact filtering configuration (uses defaults if not provided)
        metrics_prefix: Prefix for the metrics generated in this pipeline
        trajectory_grouping_hook: A hook function to override the default trajectory grouping logic

    Returns:
        Tuple of (list of TrajectoryGroups, metrics)

    Example:
        >>> from rllm.trainer.common.transform import (
        ...     transform_episodes_to_trajectory_groups,
        ... )
        >>> config = TransformConfig(
        ...     impute_missing_names=True,
        ...     drop_unnamed_traj=True,
        ...     broadcast=True,
        ... )
        >>> result = transform_episodes_to_trajectory_groups(episodes, config)
        >>> print(f"Created {len(result.trajectory_groups)} groups")
    """
    if transform_config is None:
        transform_config = TransformConfig()

    # Step 1: Name imputation
    rename_warnings = _impute_trajectory_names(episodes, transform_config)

    for warning in rename_warnings[:LOG_N_WARNINGS]:
        logger.warning(warning)

    if len(rename_warnings) > LOG_N_WARNINGS:
        logger.warning(f"Skipping {len(rename_warnings) - LOG_N_WARNINGS} more similar warnings with trajectory names")

    # Step 2: Invoke the trajectory grouping hook
    groups = trajectory_grouping_hook(episodes, transform_config, compact_filtering_config)

    # Step 3: Get metrics
    metrics = _get_transform_metrics(episodes, groups, prefix=metrics_prefix)

    return groups, metrics

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
from dataclasses import dataclass

from rllm.agents.agent import _DEFAULT_TRAJ_NAME, Episode, Trajectory, TrajectoryGroup

logger = logging.getLogger(__name__)


@dataclass
class TransformConfig:
    """Configuration for the episode-to-group transformation pipeline."""

    # Name imputation
    impute_missing_names: bool = True
    # Default trajectory name (if user does not provide a name); treated as unnamed
    default_traj_name: str = _DEFAULT_TRAJ_NAME
    # Whether to drop unnamed trajectories on failure
    drop_unnamed_traj: bool = False

    # Reward configuration
    broadcast: bool = True  # If True, use trajectory-level rewards; if False, use per-step rewards


def impute_trajectory_names(
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


def validate_and_propagate_rewards(
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
                    assert len(traj.steps) > 0, "Trajectory must have at least one step"
                    traj.reward = traj.steps[-1].reward
                    warnings.append(f"Trajectory {traj.name} in group {group.group_id} has no trajectory-level reward, propagated from last step reward")
        else:  # broadcast=False
            step_counts = [len(traj.steps) for traj in group.trajectories]
            assert len(set(step_counts)) == 1, "Trajectories in a group must have the same number of steps when broadcast=False"

    return warnings


def build_trajectory_groups(episodes: list[Episode]) -> list[TrajectoryGroup]:
    """
    Build TrajectoryGroups from episodes based on the configured grouping strategy.

    Args:
        episodes: List of episodes to group

    Returns:
        List of TrajectoryGroups
    """
    trajectories_by_name: dict[str, list[Trajectory]] = defaultdict(list)

    for episode in episodes:
        task_id = episode.id.split(":")[0]
        for trajectory in episode.trajectories:
            trajectories_by_name[f"{task_id}_{trajectory.name}"].append(trajectory)

    groups = []
    for name, trajectories in trajectories_by_name.items():
        groups.append(
            TrajectoryGroup(
                trajectories=trajectories,
                group_id=name,
            )
        )
    return groups


def transform_episodes_to_trajectory_groups(
    episodes: list[Episode],
    config: TransformConfig | None = None,
    log_n_warnings: int = 1,
) -> tuple[list[TrajectoryGroup], dict]:
    """
    Transform a list of Episodes into a list of TrajectoryGroups for advantage computation.

    This is the main entry point for the transformation pipeline. It performs:
    1. Trajectory name imputation (for unnamed trajectories)
    2. Reward validation and propagation
    3. TrajectoryGroup construction

    Args:
        episodes: List of Episodes from workflow execution
        config: Transform configuration (uses defaults if not provided)
        print_n_warnings: Number of warnings to log

    Returns:
        Tuple of (list of TrajectoryGroups, metrics)

    Example:
        >>> from rllm.trainer.common.transform import (
        ...     transform_episodes_to_trajectory_groups,
        ...     TransformConfig,
        ...     GroupingStrategy,
        ... )
        >>> config = TransformConfig(
        ...     impute_missing_names=True,
        ...     drop_unnamed_traj=True,
        ...     broadcast=True,
        ... )
        >>> result = transform_episodes_to_trajectory_groups(episodes, config)
        >>> print(f"Created {len(result.trajectory_groups)} groups")
    """
    if config is None:
        config = TransformConfig()

    metrics = dict()
    metrics["transform/num_trajs_before_transform"] = sum(len(episode.trajectories) for episode in episodes)

    # Step 1: Name imputation
    rename_warnings = impute_trajectory_names(episodes, config)

    for warning in rename_warnings[:log_n_warnings]:
        logger.warning(warning)

    if len(rename_warnings) > log_n_warnings:
        logger.warning(f"Skipping {len(rename_warnings) - log_n_warnings} more similar warnings with trajectory names")

    # Step 2: Build trajectory groups
    groups = build_trajectory_groups(episodes)

    # Step 3: Validate and propagate rewards
    reward_warnings = validate_and_propagate_rewards(groups, config)

    for warning in reward_warnings[:log_n_warnings]:
        logger.warning(warning)

    if len(reward_warnings) > log_n_warnings:
        logger.warning(f"Skipping {len(reward_warnings) - log_n_warnings} more similar warnings with reward validation")

    metrics["transform/num_trajs_after_transform"] = sum(len(group.trajectories) for group in groups)

    return groups, metrics

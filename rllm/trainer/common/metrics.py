"""
Common metric utilities for rLLM. Work with TrajectoryGroups and Episodes.
For backend-dependent metrics, please implement them in the backend-specific modules.
TODO(listar2000): think hard about what are the actually important metrics in agentic RL settings.
"""

import numpy as np

from rllm.agents.agent import TrajectoryGroup


def reduce_reward_metrics_by_trajectory_name(trajectory_groups: list[TrajectoryGroup], prefix: str = "reward") -> dict:
    """
    Reduce reward metrics by trajectory name.

    Args:
        trajectory_groups: List of TrajectoryGroup objects
        prefix: Prefix for the metric keys

    Returns:
        Dictionary of metrics by trajectory name
    """
    metrics, rewards_by_traj_name = {}, {}
    for group in trajectory_groups:
        for traj in group.trajectories:
            rewards_by_traj_name[traj.name] = traj.reward

    for traj_name, rewards in rewards_by_traj_name.items():
        metrics[f"{prefix}/{traj_name}/mean"] = np.mean(rewards)
        metrics[f"{prefix}/{traj_name}/max"] = np.max(rewards)
        metrics[f"{prefix}/{traj_name}/min"] = np.min(rewards)
        metrics[f"{prefix}/{traj_name}/std"] = np.std(rewards)

    return metrics

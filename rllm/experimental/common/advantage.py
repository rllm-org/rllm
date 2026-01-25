"""
Generic advantage computation algorithms and utilities that work on TrajectoryGroups.
"""

import logging
from collections import defaultdict
from functools import partial

import numpy as np

from rllm.agents.agent import TrajectoryGroup
from rllm.experimental.common.config import AlgorithmConfig, rLLMAdvantageEstimator

logger = logging.getLogger(__name__)


def _calculate_grpo_advantages(rewards: np.ndarray, norm_adv_by_std_in_grpo=True) -> np.ndarray:
    if rewards is None or len(rewards) < 1:
        return np.array([])
    elif len(rewards) == 1:
        group_mean, group_std = 0.0, 1.0
    else:
        group_mean = np.mean(rewards)
        group_std = np.std(rewards)

    if norm_adv_by_std_in_grpo and group_std > 1e-8:
        advantages = (rewards - group_mean) / group_std
    else:
        advantages = rewards - group_mean
    return advantages


def _calculate_reinforce_advantages(rewards: np.ndarray) -> np.ndarray:
    """REINFORCE: advantage = reward (no baseline)"""
    return rewards


def collect_reward_and_advantage_from_trajectory_groups(
    groups: list[TrajectoryGroup],
    algorithm_config: AlgorithmConfig,
    collect_advantage: bool = True,
) -> dict:
    """
    Collect reward and advantage from trajectory groups. Return a dictionary of metrics.
    If collect_advantage is False, only collect rewards.

    Args:
        groups: List of TrajectoryGroup objects
        algorithm_config: Algorithm configuration
        collect_advantage: Whether to collect advantage

    Returns:
        Dictionary of metrics
    """
    if collect_advantage:
        if algorithm_config.estimator == rLLMAdvantageEstimator.GRPO:
            advantage_fn = partial(_calculate_grpo_advantages, norm_adv_by_std_in_grpo=algorithm_config.norm_adv_by_std_in_grpo)
        elif algorithm_config.estimator == rLLMAdvantageEstimator.REINFORCE:
            advantage_fn = _calculate_reinforce_advantages
        else:
            logger.warning(f"Unsupported estimator {algorithm_config.estimator} in rLLMAdvantageEstimator, using GRPO")
            advantage_fn = partial(_calculate_grpo_advantages, norm_adv_by_std_in_grpo=algorithm_config.norm_adv_by_std_in_grpo)

        advantages_by_group = defaultdict(list)

    rewards_by_group = defaultdict(list)
    # TODO(listar2000): in the future, we should support per-trajectory-group advantage modes
    for group in groups:
        # extract the role of the group (e.g. "solver" or "judge") or assign the default name
        group_role = group.group_id.split(":")[1] if ":" in group.group_id[:-1] else "all_groups"
        if algorithm_config.stepwise_advantage_mode == "broadcast":
            assert all(traj.reward is not None for traj in group.trajectories), "Trajectory reward cannot be None in broadcast mode"
            traj_rewards = np.array([traj.reward for traj in group.trajectories])
            rewards_by_group[group_role].extend(traj_rewards)

            if collect_advantage:
                advantages = advantage_fn(traj_rewards)
                advantages_by_group[group_role].extend(advantages)
                # broadcast the advantage to all steps in the trajectory
                for traj, advantage in zip(group.trajectories, advantages, strict=False):
                    for step in traj.steps:
                        step.advantage = advantage

        elif algorithm_config.stepwise_advantage_mode == "per_step":
            assert len(set([len(traj.steps) for traj in group.trajectories])) == 1, "All trajectories must have the same number of steps in per_step mode"
            # compute advantage step by step for all trajectories
            for step_idx in range(len(group.trajectories[0].steps)):
                steps = [traj.steps[step_idx] for traj in group.trajectories]
                step_rewards = np.array([step.reward for step in steps])
                rewards_by_group[f"{group_role}_step_{step_idx}"].extend(step_rewards)
                if collect_advantage:
                    advantages = advantage_fn(step_rewards)
                    advantages_by_group[f"{group_role}_step_{step_idx}"].extend(advantages)
                    for step, advantage in zip(steps, advantages, strict=False):
                        step.advantage = advantage

    # reduce metrics by group
    final_metrics = {}
    for group_role, rewards in rewards_by_group.items():
        final_metrics[f"reward/{group_role}/mean"] = np.mean(rewards)
        final_metrics[f"reward/{group_role}/std"] = np.std(rewards)
        final_metrics[f"reward/{group_role}/max"] = np.max(rewards)
        final_metrics[f"reward/{group_role}/min"] = np.min(rewards)

    if collect_advantage:
        for group_role, advantages in advantages_by_group.items():
            final_metrics[f"advantage/{group_role}/mean"] = np.mean(advantages)
            final_metrics[f"advantage/{group_role}/std"] = np.std(advantages)
            final_metrics[f"advantage/{group_role}/max"] = np.max(advantages)
            final_metrics[f"advantage/{group_role}/min"] = np.min(advantages)
            final_metrics[f"advantage/{group_role}/fraction_zero"] = np.sum(np.abs(advantages) < 1e-8) / len(advantages)

    return final_metrics

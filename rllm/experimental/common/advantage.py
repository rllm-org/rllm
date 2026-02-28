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


def _collect_precomputed_advantages(group: TrajectoryGroup, group_role: str) -> list[float]:
    """Collect pre-computed per-token advantages from all steps.

    Called when use_precomputed_advantage is True. Steps with None or length-mismatched
    advantages are defaulted to zero lists. Raises if step.advantage is a scalar float
    (pre-computed advantages must be per-token lists).
    """
    flattened_advantages = []
    steps_missing = 0
    total_steps = 0

    for traj in group.trajectories:
        for step in traj.steps:
            total_steps += 1
            if step.advantage is None:
                step.advantage = [0.0] * len(step.response_ids)
                flattened_advantages.extend(step.advantage)
                steps_missing += 1
            elif isinstance(step.advantage, list):
                if len(step.advantage) != len(step.response_ids):
                    logger.warning(f"[group={group_role}] Step has advantage length {len(step.advantage)} but response_ids length {len(step.response_ids)}. Defaulting to zeros.")
                    step.advantage = [0.0] * len(step.response_ids)
                    steps_missing += 1
                flattened_advantages.extend(step.advantage)
            else:
                raise ValueError(f"[group={group_role}] step.advantage must be a list when use_precomputed_advantage is True, got {type(step.advantage)}")

    if steps_missing > 0:
        logger.warning(f"[group={group_role}] {steps_missing}/{total_steps} steps missing pre-computed advantages, defaulted to zeros.")

    return flattened_advantages


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
    assert algorithm_config.stepwise_advantage_mode == "broadcast", "Only broadcast mode is supported in experimental unified trainer."

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
        group_role = group.group_role

        if algorithm_config.use_precomputed_advantage:
            # Distillation mode: always use pre-computed per-token advantages from the workflow.
            if collect_advantage:
                flattened_advantages = _collect_precomputed_advantages(group, group_role)
                advantages_by_group[group_role].extend(flattened_advantages)
        else:
            # RL mode: compute advantages from trajectory rewards.
            if collect_advantage:
                # Warn if steps have pre-computed advantages that will be overwritten.
                has_any = any(step.advantage is not None for traj in group.trajectories for step in traj.steps)
                if has_any:
                    logger.warning(f"[group={group_role}] Steps have pre-computed advantages but use_precomputed_advantage is False. Overwriting with {algorithm_config.estimator.value}.")

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

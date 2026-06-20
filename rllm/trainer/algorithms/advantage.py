"""
Generic advantage computation algorithms and utilities that work on TrajectoryGroups.

Each advantage estimator will return a tuple of (advantages, returns).
"""

import logging
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from rllm.trainer.algorithms.config import AlgorithmConfig, rLLMAdvantageEstimator
from rllm.trainer.algorithms.rl_algo import calculate_grpo_advantages_per_group, calculate_rloo_advantages_per_group
from rllm.types import TrajectoryGroup
from rllm.utils.logging import DuplicateLoggingFilter

logger = logging.getLogger(__name__)
logger.addFilter(DuplicateLoggingFilter())  # prevent duplicate logging messages


RLLM_ADV_ESTIMATOR_REGISTRY: dict[str, Callable] = {}


def register_rllm_adv_estimator(name: str | rLLMAdvantageEstimator) -> Callable:
    """Register a rLLM advantage estimator — either built-in or custom.

    Registered estimators must follow the canonical signature:

        def my_estimator(
            rewards: list[np.ndarray],
            algorithm_config: AlgorithmConfig,
            **kwargs,
        ) -> tuple[list[np.ndarray], list[np.ndarray]]

    `rewards` is one entry per `TrajectoryGroup` of the same `group_role`;
    each entry is a 1-D array of scalar trajectory rewards. The output
    `(advantages_by_group, returns_by_group)` must be aligned with
    `rewards` (same outer length and same inner shapes).

    `algorithm_config` is the resolved `AlgorithmConfig`; pull whatever
    config the estimator needs (e.g. `norm_adv_by_std_in_grpo`).

    `**kwargs` carries optional per-call data injected by
    `collect_reward_and_advantage_from_trajectory_groups`. The orchestrator
    currently injects:

    * `traj_groups: list[TrajectoryGroup]` — aligned with `rewards`,
      so estimators can read per-trajectory metadata (response lengths,
      step counts, etc.) from `traj_groups[i].trajectories[j].steps`.

    Args:
        name: Name of the advantage estimator.
    """

    def decorator(func: Callable) -> Callable:
        RLLM_ADV_ESTIMATOR_REGISTRY[name] = func
        return func

    return decorator


def get_rllm_adv_estimator(name: str | rLLMAdvantageEstimator) -> Callable:
    """Get a rLLM advantage estimator by name.

    Args:
        name: Name of the advantage estimator.
    """
    if name not in RLLM_ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator {name}. If you have a custom advantage estimator, please register it using `register_rllm_adv_estimator`.")
    return RLLM_ADV_ESTIMATOR_REGISTRY[name]


@register_rllm_adv_estimator(rLLMAdvantageEstimator.GRPO)
def calculate_grpo_advantages(rewards: list[np.ndarray], algorithm_config: AlgorithmConfig, **kwargs) -> tuple[list[np.ndarray], list[np.ndarray]]:
    norm_adv_by_std_in_grpo = algorithm_config.norm_adv_by_std_in_grpo
    advantages_by_group, returns_by_group = zip(
        *[calculate_grpo_advantages_per_group(group_rewards, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo) for group_rewards in rewards],
        strict=True,
    )

    return advantages_by_group, returns_by_group


@register_rllm_adv_estimator(rLLMAdvantageEstimator.REINFORCE)
def calculate_reinforce_advantages(rewards: list[np.ndarray], algorithm_config: AlgorithmConfig, **kwargs) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """REINFORCE: advantage = reward (no baseline)"""
    return rewards, rewards


@register_rllm_adv_estimator(rLLMAdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE)
def calculate_reinforce_plus_plus_baseline_advantages(rewards: list[np.ndarray], algorithm_config: AlgorithmConfig, epsilon: float = 1e-6, **kwargs) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """REINFORCE++ baseline estimator.

    In line with Verl's REINFORCE++ baseline logic for grouped rollouts:
    1. Use per-group mean baseline when group size > 1, else baseline = 0.
    2. Whiten centered scores using role-level batch statistics.
    """
    if len(rewards) == 0:
        return [], []

    centered_rewards_by_group: list[np.ndarray] = []
    for group_rewards in rewards:
        centered_rewards_by_group.append(group_rewards - np.mean(group_rewards))

    all_centered_rewards = np.concatenate(centered_rewards_by_group)
    batch_std = np.std(all_centered_rewards)

    advantages_by_group = [centered_rewards / (batch_std + epsilon) for centered_rewards in centered_rewards_by_group]

    return advantages_by_group, advantages_by_group


@register_rllm_adv_estimator(rLLMAdvantageEstimator.PRPO)
def calculate_prpo_advantages(rewards: list[np.ndarray], algorithm_config: AlgorithmConfig, epsilon: float = 1e-6, **kwargs) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """PRPO advantage estimator, centering and normalizing rewards across the batch. See https://rllm-project.com/post.html?post=continual_learning.md

    This implementation flattens all rewards across groups, then centers by batch mean and normalizes by batch std.
    """
    if len(rewards) == 0:
        return [], []

    all_rewards = np.concatenate(rewards)
    batch_mean = np.mean(all_rewards)
    batch_std = np.std(all_rewards)

    advantages_by_group = [(group_rewards - batch_mean) / (batch_std + epsilon) for group_rewards in rewards]

    return advantages_by_group, advantages_by_group


@register_rllm_adv_estimator(rLLMAdvantageEstimator.RLOO)
def calculate_rloo_advantages(rewards: list[np.ndarray], algorithm_config: AlgorithmConfig, **kwargs) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Reinforce Leave-one-out (RLOO): https://arxiv.org/abs/2402.14740"""
    advantages_by_group, returns_by_group = zip(*[calculate_rloo_advantages_per_group(group_rewards) for group_rewards in rewards], strict=True)
    return advantages_by_group, returns_by_group


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
            if isinstance(step.advantage, float):
                step.advantage = [step.advantage] * len(step.response_ids)
            elif isinstance(step.advantage, list):
                if len(step.advantage) != len(step.response_ids):
                    logger.warning(f"[group={group_role}] Step has advantage length {len(step.advantage)} but response_ids length {len(step.response_ids)}. Defaulting to zeros.")
                    step.advantage = [0.0] * len(step.response_ids)
                    steps_missing += 1
            else:
                raise ValueError(f"[group={group_role}] step.advantage must be a scalar or a list when use_precomputed_advantage is True, got {type(step.advantage)}")

            flattened_advantages.extend(step.advantage)

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

    advantages_by_role = defaultdict(list)
    rewards_by_role = defaultdict(list)
    traj_rewards_by_role = defaultdict(list)
    traj_groups_by_role = defaultdict(list)

    for group in groups:
        group_role = group.group_role
        has_precomputed_advantage = any(step.advantage is not None for traj in group.trajectories for step in traj.steps)

        if has_precomputed_advantage and algorithm_config.use_precomputed_advantage:
            # Precompute mode (e.g. OPD, SFT): always use pre-computed per-token advantages from the workflow.
            if collect_advantage:
                flattened_advantages = _collect_precomputed_advantages(group, group_role)
                advantages_by_role[group_role].extend(flattened_advantages)
        else:
            # RL mode: compute advantages from trajectory rewards.
            if collect_advantage and has_precomputed_advantage:
                logger.warning(f"[group={group_role}] Steps have pre-computed advantages but use_precomputed_advantage is False. Overwriting with {algorithm_config.estimator.value}.")

            assert all(traj.reward is not None for traj in group.trajectories), "Trajectory reward cannot be None in broadcast mode"
            traj_rewards = np.array([traj.reward for traj in group.trajectories])
            rewards_by_role[group_role].extend(traj_rewards)

            if collect_advantage:
                traj_groups_by_role[group_role].append(group)
                traj_rewards_by_role[group_role].append(traj_rewards)

    if collect_advantage:
        for group_role, traj_groups in traj_groups_by_role.items():
            advantage_fn = get_rllm_adv_estimator(algorithm_config.estimator_map.get(group_role, algorithm_config.estimator))
            traj_rewards = traj_rewards_by_role[group_role]
            advantages_by_group, _ = advantage_fn(  # ignore returns here
                rewards=traj_rewards,
                algorithm_config=algorithm_config,
                traj_groups=traj_groups,
            )
            assert len(advantages_by_group) == len(traj_groups), "length mismatch between advantages and trajectory groups"
            for traj_group, advantages_by_traj in zip(traj_groups, advantages_by_group, strict=True):
                assert len(advantages_by_traj) == len(traj_group.trajectories), "length mismatch between trajectory rewards and computed advantages"
                advantages_by_role[group_role].extend(np.asarray(advantages_by_traj).tolist())  # for metrics calculation
                for traj, advantage in zip(traj_group.trajectories, advantages_by_traj, strict=True):
                    for step in traj.steps:
                        step.advantage = float(advantage)

    # reduce metrics by group
    final_metrics = {}
    for group_role, rewards in rewards_by_role.items():
        final_metrics[f"reward/{group_role}/mean"] = np.mean(rewards)
        final_metrics[f"reward/{group_role}/std"] = np.std(rewards)
        final_metrics[f"reward/{group_role}/max"] = np.max(rewards)
        final_metrics[f"reward/{group_role}/min"] = np.min(rewards)

    if collect_advantage:
        for group_role, advantages in advantages_by_role.items():
            final_metrics[f"advantage/{group_role}/mean"] = np.mean(advantages)
            final_metrics[f"advantage/{group_role}/std"] = np.std(advantages)
            final_metrics[f"advantage/{group_role}/max"] = np.max(advantages)
            final_metrics[f"advantage/{group_role}/min"] = np.min(advantages)
            final_metrics[f"advantage/{group_role}/fraction_zero"] = np.sum(np.abs(advantages) < 1e-8) / len(advantages)

    # Per-group difficulty / learning-signal diagnostics (keyed by role to match
    # the reward/* and advantage/* conventions above; single-agent runs emit a
    # single "default" role). Gated on `collect_advantage` because these metrics
    # describe the advantage signal (zero variance -> zero advantage); they are
    # meaningless on the validation path where no advantages are computed.
    #
    # GRPO zeros the advantage of any group whose trajectory rewards have zero
    # variance (advantage = (r - mean) / std). advantage/*/fraction_zero alone
    # can't tell *why* a group is wasted: all-solved (too easy) vs all-failed
    # (too hard). We decompose zero-variance groups by their mean reward, with
    # boundaries at the reward extremes 1.0 / 0.0; >=/<= keeps any out-of-range
    # rewards bucketed sensibly rather than silently dropped.
    #
    # The percentile metrics are two-stage, computed over the groups of a role:
    #   group_reward_mean/pXX  -- pXX over each group's *mean* reward (task difficulty spread)
    #   group_reward_std/pXX   -- pXX over each group's *within-group std* (signal magnitude spread)
    if collect_advantage:
        # Reuse `traj_rewards_by_role` (populated above for the advantage estimator)
        # so the diagnostic and the advantage path always see the same per-group
        # reward arrays. This also implicitly skips precomputed-advantage groups,
        # matching the behavior of the `reward/*` block above.
        for role, role_traj_rewards in traj_rewards_by_role.items():
            group_means: list[float] = []
            group_stds: list[float] = []
            n_total = n_informative = n_too_easy = n_too_hard = 0
            for rewards_arr in role_traj_rewards:
                # Within-group reward variance is only meaningful with >=2 trajectories.
                # A size-1 group has artifactual zero variance, not a real difficulty
                # signal -- excludes any group that rejection sampling trimmed to a
                # single trajectory.
                if len(rewards_arr) < 2:
                    continue
                mean_r = float(rewards_arr.mean())
                std_r = float(rewards_arr.std())
                group_means.append(mean_r)
                group_stds.append(std_r)
                n_total += 1
                if std_r >= 1e-8:
                    n_informative += 1  # nonzero variance -> produces gradient
                elif mean_r >= 1.0:
                    n_too_easy += 1
                elif mean_r <= 0.0:
                    n_too_hard += 1
                # else: zero-variance group stuck at a partial reward -- derivable
                # from 1 - effective - too_easy - too_hard, so not logged.

            if n_total == 0:
                continue
            # Prefixed `batch/{role}/` to sit in the same family as the existing
            # per-batch difficulty metrics (batch/solve_*, batch/steps_per_traj/*),
            # but role-keyed for multi-agent correctness (single-agent -> "default").
            final_metrics[f"batch/{role}/total"] = n_total
            final_metrics[f"batch/{role}/informative"] = n_informative
            final_metrics[f"batch/{role}/fractions/effective"] = n_informative / n_total
            final_metrics[f"batch/{role}/fractions/too_easy"] = n_too_easy / n_total
            final_metrics[f"batch/{role}/fractions/too_hard"] = n_too_hard / n_total
            means_arr = np.asarray(group_means, dtype=float)
            stds_arr = np.asarray(group_stds, dtype=float)
            for p in (10, 50, 90):
                final_metrics[f"batch/{role}/group_reward_mean/p{p}"] = float(np.percentile(means_arr, p))
                final_metrics[f"batch/{role}/group_reward_std/p{p}"] = float(np.percentile(stds_arr, p))

    return final_metrics

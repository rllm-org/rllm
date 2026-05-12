"""Preference-pair construction utilities for strict DPO training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import fmean

from rllm.types import Trajectory, TrajectoryGroup


class DPOPairingStrategy(str, Enum):
    """Supported pairing strategies for DPO preference construction."""

    BEST_WORST = "best_worst"


@dataclass
class DPOConfig:
    """Configuration for strict, on-policy DPO pair construction."""

    beta: float = 0.1
    pairing_strategy: DPOPairingStrategy | str = DPOPairingStrategy.BEST_WORST
    min_reward_gap: float = 0.0
    drop_ties: bool = True

    def __post_init__(self) -> None:
        self.pairing_strategy = DPOPairingStrategy(self.pairing_strategy)
        if self.beta <= 0:
            raise ValueError(f"DPO beta must be positive, got {self.beta}")
        if self.min_reward_gap < 0:
            raise ValueError(f"DPO min_reward_gap must be non-negative, got {self.min_reward_gap}")


@dataclass
class PreferencePair:
    """A strict chosen/rejected pair built from one trajectory group."""

    group_id: str
    task_id: str
    role: str
    chosen: Trajectory
    rejected: Trajectory
    reward_gap: float


def _mean_or_zero(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def build_preference_pairs(groups: list[TrajectoryGroup], dpo_config: DPOConfig) -> tuple[list[PreferencePair], dict[str, float | int]]:
    """Build strict DPO pairs from filtered trajectory groups.

    The current implementation intentionally supports only the simplest, safest
    shape:

    - one chosen and one rejected trajectory per group
    - only the ``best_worst`` strategy
    - only single-step trajectories
    - identical prompt token IDs across the pair
    """

    if dpo_config.pairing_strategy != DPOPairingStrategy.BEST_WORST:
        raise NotImplementedError(f"Unsupported DPO pairing strategy: {dpo_config.pairing_strategy}")

    pairs: list[PreferencePair] = []
    reward_gaps: list[float] = []
    chosen_rewards: list[float] = []
    rejected_rewards: list[float] = []

    metrics: dict[str, float | int] = {
        "dpo/groups_seen": len(groups),
        "dpo/groups_skipped_insufficient_size": 0,
        "dpo/groups_skipped_multistep": 0,
        "dpo/groups_skipped_prompt_mismatch": 0,
        "dpo/groups_skipped_tie": 0,
        "dpo/groups_skipped_small_gap": 0,
        "dpo/pairs_built": 0,
    }

    for group in groups:
        if len(group.trajectories) < 2:
            metrics["dpo/groups_skipped_insufficient_size"] += 1
            continue

        if any(len(traj.steps) != 1 for traj in group.trajectories):
            metrics["dpo/groups_skipped_multistep"] += 1
            continue

        if any(traj.reward is None for traj in group.trajectories):
            raise ValueError(f"TrajectoryGroup '{group.group_id}' has missing trajectory rewards after reward propagation.")

        ranked = sorted(group.trajectories, key=lambda traj: float(traj.reward), reverse=True)
        chosen = ranked[0]
        rejected = ranked[-1]
        chosen_reward = float(chosen.reward)
        rejected_reward = float(rejected.reward)

        if dpo_config.drop_ties and chosen_reward == rejected_reward:
            metrics["dpo/groups_skipped_tie"] += 1
            continue

        reward_gap = chosen_reward - rejected_reward
        if reward_gap < dpo_config.min_reward_gap:
            metrics["dpo/groups_skipped_small_gap"] += 1
            continue

        chosen_prompt_ids = chosen.steps[0].prompt_ids
        rejected_prompt_ids = rejected.steps[0].prompt_ids
        if chosen_prompt_ids != rejected_prompt_ids:
            metrics["dpo/groups_skipped_prompt_mismatch"] += 1
            continue

        pairs.append(
            PreferencePair(
                group_id=group.group_id,
                task_id=group.task_id,
                role=group.group_role,
                chosen=chosen,
                rejected=rejected,
                reward_gap=reward_gap,
            )
        )
        reward_gaps.append(reward_gap)
        chosen_rewards.append(chosen_reward)
        rejected_rewards.append(rejected_reward)

    metrics["dpo/pairs_built"] = len(pairs)
    metrics["dpo/reward_gap_mean"] = _mean_or_zero(reward_gaps)
    metrics["dpo/chosen_reward_mean"] = _mean_or_zero(chosen_rewards)
    metrics["dpo/rejected_reward_mean"] = _mean_or_zero(rejected_rewards)

    return pairs, metrics

"""Metrics utilities for the SkyRL backend.

Follows the same pattern as tinker_metrics_utils.py: focused, composable
functions orchestrated by a single ``update_training_metrics`` entry-point
that is called from ``on_batch_end()``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from rllm.experimental.common.metrics import reduce_metrics_by_trajectory_name

if TYPE_CHECKING:
    from rllm.experimental.unified_trainer import TrainerState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _remap_skyrl_key(key: str) -> str:
    """Map a SkyRL metric key to the rLLM logger format.

    SkyRL uses prefixes like ``policy/*``, ``critic/*``, ``loss/*``,
    ``trainer/*``.  rLLM expects ``training/policy/*``, etc.  Reward metrics
    are kept as-is.
    """
    if key.startswith(("policy/", "critic/", "loss/")):
        return f"training/{key}"
    if key.startswith("trainer/"):
        return key.replace("trainer/", "training/", 1)
    if key.startswith("reward/"):
        return key
    return f"training/{key}"


def _to_scalar(value: Any) -> float | int | None:
    """Convert *value* to a Python scalar, or return ``None`` on failure."""
    if isinstance(value, int | float):
        return value
    if hasattr(value, "item"):  # torch.Tensor
        try:
            return value.item()
        except (ValueError, RuntimeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Composable metric extractors
# ---------------------------------------------------------------------------


def extract_skyrl_training_metrics(all_metrics: dict) -> dict[str, float | int]:
    """Iterate *all_metrics*, convert values to scalars & remap keys."""
    result: dict[str, float | int] = {}
    for key, value in all_metrics.items():
        if value is None:
            continue
        scalar = _to_scalar(value)
        if scalar is None:
            continue
        result[_remap_skyrl_key(key)] = scalar
    return result


def compute_reward_metrics(trajectory_groups: list) -> dict[str, float]:
    """Compute per-trajectory-name and overall reward statistics."""
    metrics: dict[str, float] = {}

    # Per-trajectory-name breakdown
    metrics.update(reduce_metrics_by_trajectory_name(trajectory_groups, prefix="reward"))

    # Overall reward stats
    all_rewards: list[float] = []
    for group in trajectory_groups:
        for traj in group.trajectories:
            if traj.reward is not None:
                all_rewards.append(traj.reward)

    if all_rewards:
        metrics["reward/mean"] = float(np.mean(all_rewards))
        metrics["reward/max"] = float(np.max(all_rewards))
        metrics["reward/min"] = float(np.min(all_rewards))
        metrics["reward/std"] = float(np.std(all_rewards))

    return metrics


def extract_learning_rate(backend: Any) -> float | None:
    """Return the current learning rate from *backend*'s policy optimizer."""
    if hasattr(backend, "policy_model") and hasattr(backend.policy_model, "optimizer"):
        optimizer = backend.policy_model.optimizer
        if optimizer is not None and len(optimizer.param_groups) > 0:
            return optimizer.param_groups[0].get("lr", None)
    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def update_training_metrics(
    trainer_state: TrainerState,
    all_metrics: dict,
    backend: Any,
) -> None:
    """Compute and store all training metrics into *trainer_state.metrics*.

    Called once per batch from ``on_batch_end()``, after all pipeline stages
    have populated ``all_metrics`` and ``trainer_state.trajectory_groups``.
    """
    metrics = trainer_state.metrics

    # 1. SkyRL training metrics (policy/critic/loss from train_critic_and_policy, etc.)
    metrics.update(extract_skyrl_training_metrics(all_metrics))

    # 2. Reward metrics from trajectory groups
    if hasattr(trainer_state, "trajectory_groups") and trainer_state.trajectory_groups:
        metrics.update(compute_reward_metrics(trainer_state.trajectory_groups))

    # 3. Learning rate
    lr = extract_learning_rate(backend)
    if lr is not None:
        metrics["optim/lr"] = lr

    # 4. Progress metrics
    metrics["training/global_step"] = trainer_state.global_step
    metrics["training/epoch"] = trainer_state.epoch

    # 5. Debug summary
    num_training_metrics = len([k for k in metrics if k.startswith(("training/", "reward/", "optim/"))])
    logger.info(f"Step {trainer_state.global_step}: Extracted {num_training_metrics} training metrics. Keys: {sorted(metrics.keys())}")

    if len(metrics) <= 3:  # Only global_step, epoch, and maybe lr
        logger.warning(f"Step {trainer_state.global_step}: Very few metrics found. all_metrics keys were: {list(all_metrics.keys()) if all_metrics else 'empty'}. trainer_state.metrics: {list(metrics.keys())}")

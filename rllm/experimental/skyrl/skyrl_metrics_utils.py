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


def compute_progress_metrics(trainer_state: TrainerState) -> dict[str, float | int]:
    """Return Tinker-like progress metrics."""
    metrics: dict[str, float | int] = {
        "progress/batch": trainer_state.global_step,
        "progress/epoch": trainer_state.epoch,
        # Keep the existing aliases for compatibility with other backends.
        "training/global_step": trainer_state.global_step,
        "training/epoch": trainer_state.epoch,
    }
    if trainer_state.total_steps > 0:
        metrics["progress/done_frac"] = (trainer_state.global_step + 1) / trainer_state.total_steps
    return metrics


def compute_timing_metrics(trainer_state: TrainerState) -> dict[str, float]:
    """Return Tinker-compatible timing metrics."""
    return {f"time/{key}": value for key, value in trainer_state.timing_dict.items()}


def compute_env_metrics_from_trajectory_groups(trajectory_groups: list) -> dict[str, float]:
    """Compute env/all/* metrics from trajectory groups (Tinker-compatible).

    Data is available in trajectory_groups; this mirrors Tinker's compute_env_metrics
    but works with TrajectoryGroup instead of Episode.
    """
    all_rewards: list[float] = []
    prompt_token_counts: list[int] = []
    response_token_counts: list[int] = []
    total_trajectories = 0
    total_steps = 0
    ac_tokens_per_trajectory: list[int] = []
    ob_tokens_per_trajectory: list[int] = []
    episode_stats = {"all_good": 0, "all_bad": 0, "mixed": 0}
    good_threshold = 0.5
    all_step_metrics: list[dict] = []

    for group in trajectory_groups:
        group_rewards = [traj.reward if traj.reward is not None else 0.0 for traj in group.trajectories]
        all_rewards.extend(group_rewards)

        for traj in group.trajectories:
            total_trajectories += 1
            traj_ob_tokens = 0
            traj_ac_tokens = 0
            for step in traj.steps:
                ob = len(getattr(step, "prompt_ids", None) or [])
                ac = len(getattr(step, "response_ids", None) or [])
                if ob == 0 and hasattr(step, "model_output") and step.model_output is not None:
                    ob = len(getattr(step.model_output, "prompt_ids", None) or [])
                if ac == 0 and hasattr(step, "model_output") and step.model_output is not None:
                    ac = len(getattr(step.model_output, "completion_ids", None) or [])
                prompt_token_counts.append(ob)
                response_token_counts.append(ac)
                traj_ob_tokens += ob
                traj_ac_tokens += ac
                total_steps += 1
                if hasattr(step, "info") and step.info:
                    all_step_metrics.append(step.info)
            ob_tokens_per_trajectory.append(traj_ob_tokens)
            ac_tokens_per_trajectory.append(traj_ac_tokens)

        unique_rewards = len(set(group_rewards))
        if unique_rewards == 1:
            if group_rewards[0] >= good_threshold:
                episode_stats["all_good"] += 1
            else:
                episode_stats["all_bad"] += 1
        else:
            episode_stats["mixed"] += 1

    n_episodes = len(trajectory_groups)
    metrics: dict[str, float] = {
        "env/all/reward/total": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "env/all/ob_tokens_per_turn": float(np.mean(prompt_token_counts)) if prompt_token_counts else 0.0,
        "env/all/ac_tokens_per_turn": float(np.mean(response_token_counts)) if response_token_counts else 0.0,
        "env/all/ob_tokens_per_trajectory": float(np.mean(ob_tokens_per_trajectory)) if ob_tokens_per_trajectory else 0.0,
        "env/all/ac_tokens_per_trajectory": float(np.mean(ac_tokens_per_trajectory)) if ac_tokens_per_trajectory else 0.0,
        "env/all/total_episodes": float(total_trajectories),
        "env/all/total_turns": float(total_steps),
        "env/all/turns_per_episode": total_steps / total_trajectories if total_trajectories > 0 else 0.0,
        "env/all/total_ob_tokens": float(sum(prompt_token_counts)),
        "env/all/total_ac_tokens": float(sum(response_token_counts)),
        "env/all/by_episode/frac_all_good": episode_stats["all_good"] / n_episodes if n_episodes > 0 else 0.0,
        "env/all/by_episode/frac_all_bad": episode_stats["all_bad"] / n_episodes if n_episodes > 0 else 0.0,
        "env/all/by_episode/frac_mixed": episode_stats["mixed"] / n_episodes if n_episodes > 0 else 0.0,
    }

    if all_step_metrics:
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        for step_metric in all_step_metrics:
            for key, value in step_metric.items():
                if isinstance(value, int | float):
                    metric_sums[key] = metric_sums.get(key, 0) + float(value)
                    metric_counts[key] = metric_counts.get(key, 0) + 1
        for key in metric_sums:
            metrics[f"env/all/{key}"] = metric_sums[key] / metric_counts[key]

    return metrics


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

    # 2. Reward and env metrics from trajectory groups
    if hasattr(trainer_state, "trajectory_groups") and trainer_state.trajectory_groups:
        metrics.update(compute_reward_metrics(trainer_state.trajectory_groups))
        metrics.update(compute_env_metrics_from_trajectory_groups(trainer_state.trajectory_groups))

    # 3. Learning rate
    lr = extract_learning_rate(backend)
    if lr is not None:
        metrics["optim/lr"] = lr

    # 4. Progress and timing metrics (match Tinker naming)
    metrics.update(compute_progress_metrics(trainer_state))
    metrics.update(compute_timing_metrics(trainer_state))

    # 5. Debug summary
    num_training_metrics = len([k for k in metrics if k.startswith(("training/", "reward/", "optim/", "progress/", "time/", "env/"))])
    logger.info(f"Step {trainer_state.global_step}: Extracted {num_training_metrics} training metrics. Keys: {sorted(metrics.keys())}")

    if len(metrics) <= 3:  # Only global_step, epoch, and maybe lr
        logger.warning(f"Step {trainer_state.global_step}: Very few metrics found. all_metrics keys were: {list(all_metrics.keys()) if all_metrics else 'empty'}. trainer_state.metrics: {list(metrics.keys())}")

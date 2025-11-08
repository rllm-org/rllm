"""Metrics and trajectory printing utilities for Tinker-based training."""

import io
import logging
from typing import Any

import numpy as np
import tinker
import torch
from tinker_cookbook.display import colorize_example

from rllm.agents.agent import Episode

logger = logging.getLogger(__name__)


def print_episodes(
    episodes: list[Episode],
    tokenizer: Any,
    num_episodes_to_print: int = 2,
):
    """
    Print sample episodes for inspection.

    Args:
        episodes: List of Episode objects
        tokenizer: Tokenizer for decoding
        num_episodes_to_print: Number of episodes to print
    """
    from rllm.trainer.tinker.tinker_data_processor import TinkerDatumBuilder

    buf = io.StringIO()

    def bprint(s: str):
        print(s, file=buf)

    for episode_idx, episode in enumerate(episodes[:num_episodes_to_print]):
        bprint(f"\n====== Episode {episode_idx} ======")

        # Select representative trajectories from this episode (up to 4)
        max_trajs_to_print = 4
        selected_inds = list(range(len(episode.trajectories)))
        if len(episode.trajectories) > max_trajs_to_print:
            rewards = [traj.reward for traj in episode.trajectories]
            # Select uniformly distributed trajectories by reward
            sorted_inds = np.argsort(rewards)
            uniform_inds = np.linspace(0, len(sorted_inds) - 1, max_trajs_to_print).astype(int)
            selected_inds = [int(sorted_inds[i]) for i in uniform_inds]
            selected_trajs = [episode.trajectories[i] for i in selected_inds]
        else:
            selected_trajs = episode.trajectories

        # Compute advantages for this episode (based on trajectory rewards)
        episode_rewards = [traj.reward for traj in episode.trajectories]
        # Simple GRPO advantage computation (no algorithm_config needed for display)
        mean_reward = np.mean(episode_rewards)
        advantages = [r - mean_reward for r in episode_rewards]

        for traj_idx, traj in enumerate(selected_trajs):
            actual_traj_idx = selected_inds[traj_idx]
            advantage = advantages[actual_traj_idx]

            bprint(f"****** trajectory idx={actual_traj_idx}, reward={traj.reward:.3g}, advantage={advantage:.3g}, steps={len(traj.steps)} ******")

            # Build datum(s) from trajectory - this handles multi-turn merging
            datums = TinkerDatumBuilder.build_datum_from_trajectory(traj, advantage)

            for datum_idx, datum in enumerate(datums):
                bprint(f"---- datum {datum_idx + 1}/{len(datums)} ----")
                bprint(colorize_example(datum, tokenizer, key="advantages"))

        bprint("====== End Episode ======")

    logger.info(buf.getvalue().rstrip())


def print_metrics_table(metrics: dict, step: int):
    """
    Print metrics as a formatted table (similar to tinker_cookbook).

    Args:
        metrics: Dictionary of metrics
        step: Current step number
    """
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Create table
        table = Table(title=f"Step {step}", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=False)
        table.add_column("Value", justify="right", style="green")

        # Sort metrics by key for consistent ordering
        sorted_metrics = sorted(metrics.items())

        for key, value in sorted_metrics:
            # Format value based on type
            if isinstance(value, float):
                value_str = f"{value:.6f}" if abs(value) < 1000 else f"{value:.2f}"
            elif isinstance(value, int):
                value_str = str(value)
            else:
                value_str = str(value)

            table.add_row(key, value_str)

        console.print(table)

    except ImportError:
        # Fallback to simple text table if rich is not available
        print(f"\nStep {step}")
        print("=" * 60)
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                value_str = f"{value:.6f}" if abs(value) < 1000 else f"{value:.2f}"
            elif isinstance(value, int):
                value_str = str(value)
            else:
                value_str = str(value)
            print(f"{key:40s} {value_str:>15s}")
        print("=" * 60)


def print_trajectories(
    trajectories: list[list[dict]],
    advantage_computer: Any,
    tokenizer: Any,
    num_groups_to_print: int = 2,
):
    """
    Print sample trajectories for inspection (similar to tinker_cookbook print_group).

    Args:
        trajectories: List of trajectory groups
        advantage_computer: Advantage computer for computing advantages
        tokenizer: Tokenizer for decoding
        num_groups_to_print: Number of trajectory groups to print
    """
    from rllm.trainer.tinker.tinker_data_processor import TinkerDatumBuilder

    buf = io.StringIO()

    def bprint(s: str):
        print(s, file=buf)

    for group_idx, group_trajectories in enumerate(trajectories[:num_groups_to_print]):
        bprint(f"\n====== Trajectory Group {group_idx} ======")

        # Select representative trajectories from this group (up to 4)
        max_trajs_to_print = 4
        selected_inds = list(range(len(group_trajectories)))
        if len(group_trajectories) > max_trajs_to_print:
            rewards = [traj["reward"] for traj in group_trajectories]
            # Select uniformly distributed trajectories by reward
            sorted_inds = np.argsort(rewards)
            uniform_inds = np.linspace(0, len(sorted_inds) - 1, max_trajs_to_print).astype(int)
            selected_inds = [int(sorted_inds[i]) for i in uniform_inds]
            group_trajectories = [group_trajectories[i] for i in selected_inds]

        # Compute advantages for this group
        group_rewards = [traj["reward"] for traj in trajectories[group_idx]]
        advantages = advantage_computer.compute(group_rewards)

        for traj_idx, traj in enumerate(group_trajectories):
            actual_traj_idx = selected_inds[traj_idx]
            bprint(f"****** trajectory idx={actual_traj_idx}, reward={traj['reward']:.3g}, advantage={advantages[actual_traj_idx]:.3g} ******")

            # Create datum for colorized display
            datum = TinkerDatumBuilder.build_datum(traj, advantages[actual_traj_idx])
            bprint("---- datum ----")
            bprint(colorize_example(datum, tokenizer, key="advantages"))

        bprint("====== End Trajectory Group ======")

    logger.info(buf.getvalue().rstrip())


def compute_env_metrics(episodes: list[Episode]) -> dict:
    """
    Compute environment-specific metrics from episodes.

    Args:
        episodes: List of Episode objects

    Returns:
        Dictionary of environment metrics
    """
    # Collect statistics
    all_rewards = []
    prompt_token_counts = []
    response_token_counts = []
    total_trajectories = 0
    total_steps = 0
    ac_tokens_per_trajectory = []  # Action tokens per trajectory
    ob_tokens_per_trajectory = []  # Observation tokens per trajectory

    # Track episode-level statistics for GRPO
    episode_stats = {"all_good": 0, "all_bad": 0, "mixed": 0}
    good_threshold = 0.5  # Threshold for considering a reward "good"

    # Collect per-step metrics (from step.info if available)
    all_step_metrics = []

    for episode in episodes:
        # Extract rewards from all trajectories in the episode
        episode_rewards = [traj.reward for traj in episode.trajectories]
        all_rewards.extend(episode_rewards)

        # Count prompt and response tokens
        for traj in episode.trajectories:
            total_trajectories += 1
            traj_ob_tokens = 0
            traj_ac_tokens = 0
            for step in traj.steps:
                prompt_token_counts.append(len(step.prompt_ids))
                response_token_counts.append(len(step.response_ids))
                traj_ob_tokens += len(step.prompt_ids)
                traj_ac_tokens += len(step.response_ids)
                total_steps += 1

                # Collect per-step metrics if available
                if hasattr(step, "info") and step.info:
                    all_step_metrics.append(step.info)

            ob_tokens_per_trajectory.append(traj_ob_tokens)
            ac_tokens_per_trajectory.append(traj_ac_tokens)

        # Analyze episode reward distribution
        unique_rewards = len(set(episode_rewards))
        if unique_rewards == 1:
            # All same reward
            if episode_rewards[0] >= good_threshold:
                episode_stats["all_good"] += 1
            else:
                episode_stats["all_bad"] += 1
        else:
            episode_stats["mixed"] += 1

    n_episodes = len(episodes)
    metrics = {
        "env/all/reward/total": np.mean(all_rewards) if all_rewards else 0,
        "env/all/ob_tokens_per_turn": np.mean(prompt_token_counts) if prompt_token_counts else 0,
        "env/all/ac_tokens_per_turn": np.mean(response_token_counts) if response_token_counts else 0,
        "env/all/ob_tokens_per_trajectory": np.mean(ob_tokens_per_trajectory) if ob_tokens_per_trajectory else 0,
        "env/all/ac_tokens_per_trajectory": np.mean(ac_tokens_per_trajectory) if ac_tokens_per_trajectory else 0,
        "env/all/total_episodes": total_trajectories,
        "env/all/total_turns": total_steps,
        "env/all/turns_per_episode": total_steps / total_trajectories if total_trajectories > 0 else 0,
        "env/all/total_ob_tokens": sum(prompt_token_counts),
        "env/all/total_ac_tokens": sum(response_token_counts),
        "env/all/by_episode/frac_all_good": episode_stats["all_good"] / n_episodes if n_episodes > 0 else 0,
        "env/all/by_episode/frac_all_bad": episode_stats["all_bad"] / n_episodes if n_episodes > 0 else 0,
        "env/all/by_episode/frac_mixed": episode_stats["mixed"] / n_episodes if n_episodes > 0 else 0,
    }

    # Aggregate per-step metrics (format, correct, etc.)
    if all_step_metrics:
        # Compute mean of each metric type
        metric_sums = {}
        metric_counts = {}
        for step_metric in all_step_metrics:
            for key, value in step_metric.items():
                if isinstance(value, int | float):
                    metric_sums[key] = metric_sums.get(key, 0) + value
                    metric_counts[key] = metric_counts.get(key, 0) + 1

        for key in metric_sums:
            metrics[f"env/all/{key}"] = metric_sums[key] / metric_counts[key]

    return metrics


def compute_kl_and_entropy_metrics(training_datums: list[tinker.Datum], training_logprobs: list[torch.Tensor]) -> dict:
    """
    Compute KL divergence and entropy metrics from training.

    Args:
        training_datums: List of training datums
        training_logprobs: List of training logprobs from forward_backward

    Returns:
        Dictionary of KL and entropy metrics
    """
    all_diffs = []
    all_sampling_logprobs = []

    for datum, training_logprobs_tensor in zip(training_datums, training_logprobs, strict=False):
        # Get logprobs from sampling
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        action_mask = datum.loss_fn_inputs["mask"].to_torch() > 0

        # Extract only action token logprobs
        sampling_logprobs_actions = sampling_logprobs[action_mask]
        training_logprobs_actions = training_logprobs_tensor[action_mask]

        if len(sampling_logprobs_actions) > 0:
            logprob_diff = sampling_logprobs_actions - training_logprobs_actions
            all_diffs.append(logprob_diff)
            all_sampling_logprobs.append(sampling_logprobs_actions)

    if not all_diffs:
        return {}

    flat_diffs = torch.cat(all_diffs)
    kl_sample_train_v1 = flat_diffs.mean().item()
    kl_sample_train_v2 = 0.5 * (flat_diffs**2).mean().item()

    flat_sampling_logprobs = torch.cat(all_sampling_logprobs)
    entropy_sample = -flat_sampling_logprobs.mean().item()

    # Compute perplexity: exp(-mean_logprob)
    mean_logprob = flat_sampling_logprobs.mean().item()
    perplexity = torch.exp(-torch.tensor(mean_logprob)).item()

    return {
        "optim/kl_sample_train_v1": kl_sample_train_v1,
        "optim/kl_sample_train_v2": kl_sample_train_v2,
        "optim/entropy": entropy_sample,
        "optim/perplexity": perplexity,
    }


def compute_training_metrics(
    episodes: list[Episode],
    batch_idx: int,
    time_metrics: dict,
    learning_rate: float,
    total_batches: int | None = None,
    epoch: int = 0,
    training_datums: list[tinker.Datum] | None = None,
    training_logprobs: list[torch.Tensor] | None = None,
) -> dict:
    """
    Compute comprehensive training metrics.

    Args:
        episodes: List of Episode objects
        batch_idx: Current batch index
        time_metrics: Dictionary of time measurements
        learning_rate: Current learning rate
        total_batches: Total number of batches (optional, for progress tracking)
        epoch: Current epoch number
        training_datums: Optional list of training datums for KL computation
        training_logprobs: Optional list of training logprobs for KL computation

    Returns:
        Dictionary of metrics
    """
    # Basic progress metrics
    metrics = {
        "progress/batch": batch_idx,
        "progress/epoch": epoch,
        "optim/lr": learning_rate,
    }

    # Add progress fraction if total batches is known
    if total_batches is not None and total_batches > 0:
        metrics["progress/done_frac"] = (batch_idx + 1) / total_batches

    # Add time metrics
    metrics.update(time_metrics)

    # Collect all rewards from Episode objects
    all_rewards = []
    for episode in episodes:
        for traj in episode.trajectories:
            all_rewards.append(traj.reward)

    if all_rewards:
        metrics.update(
            {
                "reward/mean": np.mean(all_rewards),
                "reward/max": np.max(all_rewards),
                "reward/min": np.min(all_rewards),
                "reward/std": np.std(all_rewards),
            }
        )

    # Add environment metrics (detailed stats similar to tinker_cookbook)
    env_metrics = compute_env_metrics(episodes)
    metrics.update(env_metrics)

    # Add KL and entropy metrics if available
    if training_datums is not None and training_logprobs is not None:
        kl_metrics = compute_kl_and_entropy_metrics(training_datums, training_logprobs)
        metrics.update(kl_metrics)

    return metrics

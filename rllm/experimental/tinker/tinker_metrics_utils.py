"""Metrics and trajectory printing utilities for Tinker-based training."""

import io
import logging
from typing import Any

import numpy as np
import torch
from tinker_cookbook.display import colorize_example

import tinker
from rllm.experimental.unified_trainer import TrainerState

logger = logging.getLogger(__name__)


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


def update_training_metrics(trainer_state: TrainerState, learning_rate: float, total_batches: int | None = None) -> None:
    """
    Compute comprehensive training metrics.
    Note that the advantage and reward metrics are already computed and stored in the trainer state.

    Args:
        trainer_state: TrainerState object
        learning_rate: Current learning rate
        total_batches: Total number of batches (optional, for progress tracking)
    """
    metrics = trainer_state.metrics
    # Basic progress metrics
    metrics.update(
        {
            "progress/batch": trainer_state.global_step,
            "progress/epoch": trainer_state.epoch,
            "optim/lr": learning_rate,
        }
    )

    # Add progress fraction if total batches is known
    if total_batches is not None and total_batches > 0:
        metrics["progress/done_frac"] = (trainer_state.global_step + 1) / total_batches

    # Add time metrics (adding a "time/" prefix to the keys for compatibility)
    metrics.update({f"time/{key}": value for key, value in trainer_state.timing_dict.items()})

    # Add environment metrics (detailed stats similar to tinker_cookbook)
    # TODO(listar2000): actually implement separate metrics that are episode-based (currently trajectory-group-based)
    # env_metrics = compute_env_metrics(episodes)
    # metrics.update(env_metrics)

    # Add KL and entropy metrics if available
    if "training_logprobs" in trainer_state.extra_info:
        training_datums = trainer_state.backend_batch
        training_logprobs = trainer_state.extra_info["training_logprobs"]
        metrics.update(compute_kl_and_entropy_metrics(training_datums, training_logprobs))  # type: ignore[arg-type]

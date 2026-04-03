"""Metrics and trajectory printing utilities for Tinker-based training."""

import logging

import tinker
import torch

from rllm.experimental.common.visualization import print_metrics_table  # noqa: F401 (re-export)
from rllm.experimental.unified_trainer import TrainerState

logger = logging.getLogger(__name__)



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

    # Compute perplexity: exp(entropy)
    perplexity = torch.exp(torch.tensor(entropy_sample)).item()

    return {
        "train/kl_sample_train_v1": kl_sample_train_v1,
        "train/kl_sample_train_v2": kl_sample_train_v2,
        "train/entropy": entropy_sample,
        "train/perplexity": perplexity,
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
            "progress/lr": learning_rate,
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

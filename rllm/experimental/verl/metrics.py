"""Small metric helpers shared by rLLM's Verl integrations."""

from __future__ import annotations

import torch


def compute_rollout_probs_diff_metrics(
    *,
    rollout_log_probs: torch.Tensor,
    actor_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute rollout-vs-actor probability drift over loss-bearing response tokens only."""
    rollout_probs = torch.exp(rollout_log_probs)
    actor_probs = torch.exp(actor_log_probs)
    rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
    masked_rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())

    if masked_rollout_probs_diff.numel() == 0:
        return {}

    if masked_rollout_probs_diff.numel() == 1:
        std = 0.0
    else:
        std = torch.std(masked_rollout_probs_diff).detach().item()

    return {
        "training/rollout_probs_diff_max": torch.max(masked_rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(masked_rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": std,
    }

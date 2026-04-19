import torch


def compute_rollout_probs_diff_metrics(
    rollout_old_log_probs: torch.Tensor,
    actor_old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute rollout-vs-actor probability diff metrics on assistant tokens only."""
    rollout_probs = torch.exp(rollout_old_log_probs)
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
    rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())

    return {
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
    }

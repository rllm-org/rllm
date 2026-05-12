"""DPO actor loss helpers for the Verl backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def sequence_log_probs(token_log_probs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Sum token log-probs over generated response tokens."""
    if token_log_probs.shape != response_mask.shape:
        raise ValueError(f"token_log_probs and response_mask must have the same shape, got {token_log_probs.shape} and {response_mask.shape}")
    return (token_log_probs * response_mask.to(token_log_probs.dtype)).sum(dim=-1)


def _pair_weights(row_weights: torch.Tensor | None, num_pairs: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if row_weights is None:
        return torch.ones(num_pairs, device=device, dtype=dtype)
    if row_weights.ndim > 1:
        row_weights = row_weights.reshape(row_weights.shape[0], -1)[:, 0]
    if row_weights.numel() == num_pairs:
        return row_weights.to(device=device, dtype=dtype)
    if row_weights.numel() != num_pairs * 2:
        raise ValueError(f"DPO pair weights must have {num_pairs} or {num_pairs * 2} entries, got {row_weights.numel()}")
    chosen_weights = row_weights[0::2].to(device=device, dtype=dtype)
    rejected_weights = row_weights[1::2].to(device=device, dtype=dtype)
    return torch.minimum(chosen_weights, rejected_weights)


def compute_dpo_loss(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    beta: float,
    pair_weights: torch.Tensor | None = None,
    is_chosen: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute strict pairwise DPO loss for rows ordered as chosen/rejected pairs."""
    if policy_log_probs.shape != ref_log_probs.shape:
        raise ValueError(f"policy_log_probs and ref_log_probs must have the same shape, got {policy_log_probs.shape} and {ref_log_probs.shape}")
    if policy_log_probs.shape != response_mask.shape:
        raise ValueError(f"response_mask must match log-prob shape, got {response_mask.shape} and {policy_log_probs.shape}")
    if policy_log_probs.ndim != 2:
        raise ValueError(f"DPO loss expects padded [batch, response_length] log-probs, got shape {policy_log_probs.shape}")
    if policy_log_probs.shape[0] % 2 != 0:
        raise ValueError(f"DPO loss expects an even number of rows, got {policy_log_probs.shape[0]}")
    if beta <= 0:
        raise ValueError(f"DPO beta must be positive, got {beta}")
    if is_chosen is not None:
        is_chosen = is_chosen.reshape(-1).to(torch.bool)
        if is_chosen.numel() != policy_log_probs.shape[0]:
            raise ValueError(f"is_chosen must have {policy_log_probs.shape[0]} entries, got {is_chosen.numel()}")
        if not torch.all(is_chosen[0::2]) or torch.any(is_chosen[1::2]):
            raise ValueError("DPO loss expects rows ordered as chosen/rejected pairs")

    policy_seq_logp = sequence_log_probs(policy_log_probs, response_mask)
    ref_seq_logp = sequence_log_probs(ref_log_probs, response_mask)

    chosen_policy = policy_seq_logp[0::2]
    rejected_policy = policy_seq_logp[1::2]
    chosen_ref = ref_seq_logp[0::2]
    rejected_ref = ref_seq_logp[1::2]

    chosen_margin = chosen_policy - chosen_ref
    rejected_margin = rejected_policy - rejected_ref
    margins = chosen_margin - rejected_margin
    losses = -F.logsigmoid(beta * margins)

    weights = _pair_weights(pair_weights, losses.numel(), device=losses.device, dtype=losses.dtype)
    denom = weights.sum().clamp_min(1.0)
    loss = (losses * weights).sum() / denom

    metrics = {
        "dpo/loss": loss.detach(),
        "dpo/accuracy": (((margins > 0).to(losses.dtype) * weights).sum() / denom).detach(),
        "dpo/margin_mean": ((margins * weights).sum() / denom).detach(),
        "dpo/chosen_logp_mean": ((chosen_policy * weights).sum() / denom).detach(),
        "dpo/rejected_logp_mean": ((rejected_policy * weights).sum() / denom).detach(),
    }
    return loss, metrics


@dataclass
class CustomDPOLoss:
    """Verl ``set_loss_fn`` callable for strict chosen/rejected DPO batches."""

    beta: float

    def __call__(self, model_output, data, dp_group=None):  # noqa: ANN001, ARG002
        from verl.utils.metric import AggregationType, Metric
        from verl.workers.utils.padding import no_padding_2_padding

        policy_log_probs = no_padding_2_padding(model_output["log_probs"], data)

        pair_weights = data["dpo_pair_weights"] if "dpo_pair_weights" in data else None
        is_chosen = data["dpo_is_chosen"] if "dpo_is_chosen" in data else None
        padded_data = data.select("response_mask", "ref_log_prob").to_padded_tensor()

        loss, scalar_metrics = compute_dpo_loss(
            policy_log_probs=policy_log_probs,
            ref_log_probs=padded_data["ref_log_prob"],
            response_mask=padded_data["response_mask"].to(bool),
            beta=self.beta,
            pair_weights=pair_weights,
            is_chosen=is_chosen,
        )
        metrics = {key: Metric(value=value, aggregation=AggregationType.MEAN) for key, value in scalar_metrics.items()}
        metrics["actor/pg_loss"] = Metric(value=loss, aggregation=AggregationType.MEAN)
        return loss, metrics

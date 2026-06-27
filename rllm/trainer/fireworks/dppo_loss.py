"""DPPO custom loss for the Fireworks training backend."""

from __future__ import annotations

from typing import Any, Literal

DPPODivergenceType = Literal["tv", "kl"]

SAFETY_CLAMP = 20.0
PROB_CLAMP_MIN = -30.0


def compute_binary_divergence(behavior_logprobs, policy_logprobs, response_mask, divergence_type: str):
    """Binary sampled-token divergence used by DPPO."""
    import torch

    divergence_type = str(divergence_type).lower()
    mu = torch.exp(behavior_logprobs.clamp(min=PROB_CLAMP_MIN, max=0.0))
    pi = torch.exp(policy_logprobs.clamp(min=PROB_CLAMP_MIN, max=0.0))

    if divergence_type == "tv":
        divergence = (mu - pi).abs()
    elif divergence_type == "kl":
        eps = 1e-9
        mu_clip = mu.clamp(eps, 1.0 - eps)
        pi_clip = pi.clamp(eps, 1.0 - eps)
        divergence = mu_clip * (mu_clip.log() - pi_clip.log()) + (1.0 - mu_clip) * ((1.0 - mu_clip).log() - (1.0 - pi_clip).log())
    else:
        raise ValueError("dppo_divergence_type must be 'tv' or 'kl'")

    return torch.where(response_mask, divergence, torch.zeros_like(divergence))


def compute_dppo_mask(
    policy_logprobs,
    behavior_logprobs,
    advantages,
    ratio,
    response_mask,
    divergence_type: str,
    divergence_threshold: float,
):
    """Return the detached DPPO trust-region mask and divergence."""
    import torch

    with torch.no_grad():
        divergence = compute_binary_divergence(
            behavior_logprobs=behavior_logprobs,
            policy_logprobs=policy_logprobs,
            response_mask=response_mask,
            divergence_type=divergence_type,
        )
        outside_region = divergence > divergence_threshold
        bad_high = (advantages > 0) & (ratio > 1.0) & outside_region
        bad_low = (advantages < 0) & (ratio < 1.0) & outside_region
        bad = bad_high | bad_low
        mask = (~bad & response_mask).to(policy_logprobs.dtype)
    return mask, divergence


def _tensor_data_to_tensor(value: Any, *, dtype, device, name: str, sample_idx: int):
    import torch

    if value is None:
        raise ValueError(f"DPPO custom loss datum {sample_idx} is missing '{name}'")
    if hasattr(value, "to_torch"):
        return value.to_torch().to(dtype=dtype, device=device)
    if hasattr(value, "data"):
        return torch.tensor(list(value.data), dtype=dtype, device=device)
    return torch.tensor(list(value), dtype=dtype, device=device)


def make_dppo_loss_fn(
    advantages: list[float],
    rollout_logprobs: list[list[float]],
    *,
    divergence_type: str = "tv",
    divergence_threshold: float = 0.1,
    ratio_log_cap: float = SAFETY_CLAMP,
):
    """Build a Fireworks ``forward_backward_custom`` DPPO loss closure.

    The behavior policy is the rollout/inference policy. No proximal forward
    pass is used.
    """
    divergence_type = str(divergence_type).lower()
    if divergence_type not in ("tv", "kl"):
        raise ValueError("dppo_divergence_type must be 'tv' or 'kl'")
    if divergence_threshold <= 0.0:
        raise ValueError("dppo_divergence_threshold must be > 0")

    def loss_fn(data, logprobs_list):
        import torch

        if not logprobs_list:
            return torch.tensor(0.0, requires_grad=True), {
                "dppo_mask_frac_kept": 0.0,
                "dppo_active_tokens": 0.0,
            }

        total_loss = logprobs_list[0].sum() * 0.0
        kept_tokens = 0.0
        active_tokens = 0.0
        divergence_sum = 0.0
        divergence_max = 0.0
        ratio_sum = 0.0

        for i, policy_lp_full in enumerate(logprobs_list):
            if i >= len(data):
                raise ValueError(f"DPPO custom loss got logprobs for sample {i}, but only {len(data)} datums")
            if i >= len(advantages) or i >= len(rollout_logprobs):
                raise ValueError(f"DPPO custom loss missing rollout tensors for sample {i}")

            weights = _tensor_data_to_tensor(
                data[i].loss_fn_inputs.get("weights"),
                dtype=policy_lp_full.dtype,
                device=policy_lp_full.device,
                name="weights",
                sample_idx=i,
            )
            n_tokens = int(policy_lp_full.shape[0])
            if int(weights.shape[0]) < n_tokens:
                raise ValueError(f"DPPO requires weights for all target tokens in sample {i}: need {n_tokens}, got {int(weights.shape[0])}")
            rollout_lp_values = rollout_logprobs[i]
            if len(rollout_lp_values) < n_tokens:
                raise ValueError(f"DPPO requires rollout logprobs for all target tokens in sample {i}: need {n_tokens}, got {len(rollout_lp_values)}")

            weights = weights[:n_tokens]
            policy_lp = policy_lp_full[:n_tokens]
            behavior_lp = torch.tensor(
                rollout_lp_values[:n_tokens],
                dtype=policy_lp.dtype,
                device=policy_lp.device,
            )
            response_mask = weights > 0.5
            active_count = int(response_mask.sum().item())
            if active_count == 0:
                continue

            adv = policy_lp.new_full(policy_lp.shape, float(advantages[i]))
            log_ratio = torch.clamp(policy_lp - behavior_lp, min=-ratio_log_cap, max=ratio_log_cap)
            ratio = torch.exp(log_ratio)
            dppo_mask, divergence = compute_dppo_mask(
                policy_logprobs=policy_lp,
                behavior_logprobs=behavior_lp,
                advantages=adv,
                ratio=ratio.detach(),
                response_mask=response_mask,
                divergence_type=divergence_type,
                divergence_threshold=divergence_threshold,
            )

            per_token_loss = -adv * ratio * dppo_mask
            total_loss = total_loss + per_token_loss.sum()

            active_tokens += float(active_count)
            kept_tokens += float(dppo_mask.sum().item())
            active_divergence = divergence[response_mask]
            active_ratio = ratio.detach()[response_mask]
            divergence_sum += float(active_divergence.sum().item())
            divergence_max = max(divergence_max, float(active_divergence.max().item()))
            ratio_sum += float(active_ratio.sum().item())

        denom = max(active_tokens, 1.0)
        metrics = {
            "dppo_mask_frac_kept": kept_tokens / denom,
            "dppo_active_tokens": active_tokens,
            "dppo_divergence_mean": divergence_sum / denom,
            "dppo_divergence_max": divergence_max,
            "dppo_divergence_threshold": float(divergence_threshold),
            "dppo_ratio_mean": ratio_sum / denom,
            "mean_loss": float(total_loss.detach().item()) / denom,
        }
        metrics[f"dppo_divergence_type/{divergence_type}"] = 1.0
        return total_loss, metrics

    return loss_fn

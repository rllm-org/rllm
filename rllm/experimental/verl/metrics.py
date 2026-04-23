"""Compatibility helpers around Verl's debug metric utilities."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _load_verl_calculate_debug_metrics() -> Callable[[Any], dict[str, float]]:
    """Load Verl's debug metrics helper lazily to preserve optional dependency boundaries."""
    from verl.utils.debug.metrics import calculate_debug_metrics

    return calculate_debug_metrics


def _resolve_log_prob_mask(data: Any) -> torch.Tensor:
    """Resolve the response-side mask exactly as Verl's helper does."""
    batch = data.batch
    response_length = batch["responses"].size(1)

    if response_length == 0:
        return torch.zeros_like(batch["rollout_log_probs"], dtype=torch.bool)

    if "response_mask" in batch:
        log_prob_mask = batch["response_mask"]
    elif "attention_mask" in batch:
        log_prob_mask = batch["attention_mask"]
    else:
        return torch.ones_like(batch["rollout_log_probs"], dtype=torch.bool)

    return log_prob_mask[:, -response_length:].bool()


def _default_debug_metrics() -> dict[str, float]:
    """Mirror the newer upstream fallback for all-zero valid-token masks."""
    return {
        "training/rollout_probs_diff_valid": 0,
        "training/rollout_probs_diff_max": float("nan"),
        "training/rollout_probs_diff_mean": float("nan"),
        "training/rollout_probs_diff_std": float("nan"),
        "training/rollout_actor_probs_pearson_corr": float("nan"),
    }


def calculate_debug_metrics_compat(data: Any) -> dict[str, float]:
    """Delegate to Verl's helper while backfilling newer empty-mask behavior for v0.7.1."""
    response_mask = _resolve_log_prob_mask(data)

    if not response_mask.any().item():
        logger.warning("response_mask is all False, returning default metrics")
        return _default_debug_metrics()

    metrics = _load_verl_calculate_debug_metrics()(data)

    if response_mask.sum().item() == 1 and metrics.get("training/rollout_probs_diff_valid") == 1:
        metrics["training/rollout_probs_diff_std"] = 0.0

    return metrics

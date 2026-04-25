"""Compatibility helpers around Verl's debug metric utilities."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)
_EMPTY_REDUCTION_ERROR_SNIPPETS = (
    "Expected reduction dim to be specified",
    "input.numel() == 0",
)


def _load_verl_calculate_debug_metrics() -> Callable[[Any], dict[str, float]]:
    """Load Verl's debug metrics helper lazily to preserve optional dependency boundaries."""
    from verl.utils.debug.metrics import calculate_debug_metrics

    return calculate_debug_metrics


def _default_debug_metrics() -> dict[str, float]:
    """Mirror the newer upstream fallback for all-zero valid-token masks."""
    return {
        "training/rollout_probs_diff_valid": 0,
        "training/rollout_probs_diff_max": float("nan"),
        "training/rollout_probs_diff_mean": float("nan"),
        "training/rollout_probs_diff_std": float("nan"),
        "training/rollout_actor_probs_pearson_corr": float("nan"),
    }


def _is_legacy_empty_mask_error(exc: RuntimeError) -> bool:
    """Detect the empty-mask reduction failure from older Verl helper versions."""
    message = str(exc)
    return all(snippet in message for snippet in _EMPTY_REDUCTION_ERROR_SNIPPETS)


def _normalize_degenerate_std(metrics: dict[str, float]) -> dict[str, float]:
    """Clamp the single-token std case to zero without masking broader metric failures."""
    std = metrics.get("training/rollout_probs_diff_std", float("nan"))
    if metrics.get("training/rollout_probs_diff_valid") != 1 or not math.isnan(std):
        return metrics

    if math.isnan(metrics.get("training/rollout_probs_diff_max", float("nan"))):
        return metrics
    if math.isnan(metrics.get("training/rollout_probs_diff_mean", float("nan"))):
        return metrics

    metrics["training/rollout_probs_diff_std"] = 0.0
    return metrics


def calculate_debug_metrics_compat(data: Any) -> dict[str, float]:
    """Delegate to Verl's helper while backfilling legacy empty-mask behavior."""
    try:
        metrics = _load_verl_calculate_debug_metrics()(data)
    except RuntimeError as exc:
        if not _is_legacy_empty_mask_error(exc):
            raise
        logger.warning("Verl debug metrics hit an empty valid-token mask, returning default metrics")
        return _default_debug_metrics()

    return _normalize_degenerate_std(metrics)

"""Compatibility helpers around Verl's debug metric utilities."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from verl.utils.debug.metrics import calculate_debug_metrics

if TYPE_CHECKING:
    from verl.protocol import DataProto

logger = logging.getLogger(__name__)
_EMPTY_REDUCTION_ERROR_SNIPPETS = (
    "Expected reduction dim to be specified",
    "input.numel() == 0",
)


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


def calculate_debug_metrics_compat(batch: DataProto) -> dict[str, float]:
    """Delegate to Verl's metric helpers while backfilling legacy empty-mask behavior."""
    try:
        metrics = calculate_debug_metrics(batch)
    except RuntimeError as exc:
        if not _is_legacy_empty_mask_error(exc):
            raise
        logger.warning("Verl debug metrics hit an empty valid-token mask, returning default metrics")
        return {
            "training/rollout_probs_diff_valid": 0,
            "training/rollout_probs_diff_max": float("nan"),
            "training/rollout_probs_diff_mean": float("nan"),
            "training/rollout_probs_diff_std": float("nan"),
            "training/rollout_actor_probs_pearson_corr": float("nan"),
        }

    return _normalize_degenerate_std(metrics)

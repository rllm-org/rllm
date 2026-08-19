"""MetricsAggregator for async training.

Accumulates metric observations from multiple sources (buffer, training loop,
coordinator) and reduces them with per-key aggregation rules at flush time.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

# Keys that should be summed rather than averaged.
_SUM_KEYS: set[str] = {
    "groups/num_trajs_before_filter",
    "groups/num_trajs_after_filter",
    "groups/num_groups",
    "groups/dropped_min_trajs",
    "groups/dropped_zero_adv",
}

# Prefixes where "last value" is the correct reduction.
_LAST_PREFIXES: tuple[str, ...] = (
    "time/",
    "train/",
    "progress/",
    "async/",
)

# Prefixes where "mean" is the correct reduction.
_MEAN_PREFIXES: tuple[str, ...] = ("episode/",)


def _infer_rule(key: str) -> str:
    """Infer aggregation rule from metric key name.

    Resolution order:
    1. Explicit sum keys
    2. Prefix-based rules (last or mean)
    3. Keyword-based rules in the final metric component
       (max, min, mean, avg, std, fraction)
    4. Default: mean
    """
    if key in _SUM_KEYS:
        return "sum"

    for prefix in _LAST_PREFIXES:
        if key.startswith(prefix):
            return "last"

    for prefix in _MEAN_PREFIXES:
        if key.startswith(prefix):
            return "mean"

    # Infer from the metric name, not arbitrary path components. A role such as
    # ``mini-swe-agent`` must not make every reward metric use the ``min``
    # reducer. Splitting on ``_`` preserves names such as ``max_group_size``.
    metric_tokens = set(key.rsplit("/", 1)[-1].replace(":", "_").split("_"))
    if "max" in metric_tokens:
        return "max"
    if "min" in metric_tokens:
        return "min"
    if "mean" in metric_tokens or "avg" in metric_tokens:
        return "mean"
    if "std" in metric_tokens or "fraction" in metric_tokens:
        return "mean"

    return "mean"


def _reduce(rule: str, values: list[float]) -> float:
    if rule == "mean":
        return sum(values) / len(values)
    if rule == "sum":
        return sum(values)
    if rule == "max":
        return max(values)
    if rule == "min":
        return min(values)
    if rule == "last":
        return values[-1]
    return sum(values) / len(values)


class MetricsAggregator:
    """Accumulates metric observations and flushes as an aggregated plain dict.

    Usage::

        agg = MetricsAggregator()

        # record from various sources
        agg.record("episode/queue_wait", 0.3)
        agg.record("episode/queue_wait", 0.5)
        agg.record_dict(transform_metrics)

        # at log time
        plain_dict = agg.flush()  # reduces, clears, returns dict
    """

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)

    def record(self, key: str, value: float) -> None:
        """Record a single metric observation."""
        self._values[key].append(float(value))

    def record_dict(self, metrics: dict) -> None:
        """Record all numeric values from a dict, coercing types."""
        for k, v in metrics.items():
            if isinstance(v, int | float):
                self._values[k].append(float(v))
            elif isinstance(v, np.number):
                self._values[k].append(float(v))

    def flush(self) -> dict[str, float]:
        """Reduce all accumulated values and return a plain dict. Clears state."""
        result = {}
        for key, values in self._values.items():
            if values:
                result[key] = _reduce(_infer_rule(key), values)
        self._values.clear()
        return result

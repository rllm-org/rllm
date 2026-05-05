"""Tests for small Verl-specific metric helpers."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_METRICS_PATH = Path(__file__).resolve().parents[1] / "rllm" / "experimental" / "verl" / "metrics.py"
_SPEC = importlib.util.spec_from_file_location("rllm_test_verl_metrics", _METRICS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_METRICS_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_METRICS_MODULE)
calculate_debug_metrics_compat = _METRICS_MODULE.calculate_debug_metrics_compat


class _DummyData:
    def __init__(self, batch: dict[str, object]) -> None:
        self.batch = batch


def test_calculate_debug_metrics_compat_passes_through_upstream_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normal upstream results should pass through unchanged."""
    expected_metrics = {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": 0.3,
        "training/rollout_probs_diff_mean": 0.2,
        "training/rollout_probs_diff_std": 0.1,
        "training/rollout_actor_probs_pearson_corr": 0.9,
    }
    calls: list[_DummyData] = []

    def fake_upstream(data: _DummyData) -> dict[str, float]:
        calls.append(data)
        return expected_metrics

    monkeypatch.setattr(_METRICS_MODULE, "_load_verl_calculate_debug_metrics", lambda: fake_upstream)

    data = _DummyData({"batch_id": "pass-through"})

    metrics = calculate_debug_metrics_compat(data)

    assert calls == [data]
    assert metrics == expected_metrics


def test_calculate_debug_metrics_compat_keeps_newer_upstream_empty_mask_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If upstream already returns the default empty-mask payload, keep it unchanged."""
    expected_metrics = {
        "training/rollout_probs_diff_valid": 0,
        "training/rollout_probs_diff_max": float("nan"),
        "training/rollout_probs_diff_mean": float("nan"),
        "training/rollout_probs_diff_std": float("nan"),
        "training/rollout_actor_probs_pearson_corr": float("nan"),
    }

    monkeypatch.setattr(_METRICS_MODULE, "_load_verl_calculate_debug_metrics", lambda: lambda _: dict(expected_metrics))

    metrics = calculate_debug_metrics_compat(_DummyData({"batch_id": "newer-upstream-empty-mask"}))

    assert metrics["training/rollout_probs_diff_valid"] == 0
    assert math.isnan(metrics["training/rollout_probs_diff_max"])
    assert math.isnan(metrics["training/rollout_probs_diff_mean"])
    assert math.isnan(metrics["training/rollout_probs_diff_std"])
    assert math.isnan(metrics["training/rollout_actor_probs_pearson_corr"])


def test_calculate_debug_metrics_compat_single_token_std_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single valid-token masks should normalize std to zero instead of NaN."""
    upstream_metrics = {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": 0.5,
        "training/rollout_probs_diff_mean": 0.5,
        "training/rollout_probs_diff_std": float("nan"),
        "training/rollout_actor_probs_pearson_corr": float("nan"),
    }

    monkeypatch.setattr(_METRICS_MODULE, "_load_verl_calculate_debug_metrics", lambda: lambda _: dict(upstream_metrics))

    metrics = calculate_debug_metrics_compat(_DummyData({"batch_id": "single-token"}))

    assert metrics["training/rollout_probs_diff_std"] == pytest.approx(0.0)
    assert math.isnan(metrics["training/rollout_actor_probs_pearson_corr"])


def test_calculate_debug_metrics_compat_backfills_legacy_empty_mask_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older upstream helpers raise on empty reductions; the wrapper should return default metrics."""

    def raise_empty_mask(_: _DummyData) -> dict[str, float]:
        raise RuntimeError(f"max(): {_METRICS_MODULE._EMPTY_REDUCTION_ERROR_SNIPPETS[0]} for {_METRICS_MODULE._EMPTY_REDUCTION_ERROR_SNIPPETS[1]}")

    monkeypatch.setattr(
        _METRICS_MODULE,
        "_load_verl_calculate_debug_metrics",
        lambda: raise_empty_mask,
    )

    metrics = calculate_debug_metrics_compat(_DummyData({"batch_id": "legacy-empty-mask"}))

    assert metrics["training/rollout_probs_diff_valid"] == 0
    assert math.isnan(metrics["training/rollout_probs_diff_max"])
    assert math.isnan(metrics["training/rollout_probs_diff_mean"])
    assert math.isnan(metrics["training/rollout_probs_diff_std"])
    assert math.isnan(metrics["training/rollout_actor_probs_pearson_corr"])


def test_calculate_debug_metrics_compat_reraises_unrelated_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not hide unrelated upstream failures behind the empty-mask compatibility path."""

    def raise_unrelated(_: _DummyData) -> dict[str, float]:
        raise RuntimeError("unexpected upstream failure")

    monkeypatch.setattr(_METRICS_MODULE, "_load_verl_calculate_debug_metrics", lambda: raise_unrelated)

    with pytest.raises(RuntimeError, match="unexpected upstream failure"):
        calculate_debug_metrics_compat(_DummyData({"batch_id": "unexpected-error"}))

"""Tests for small Verl-specific metric helpers."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

_METRICS_PATH = Path(__file__).resolve().parents[1] / "rllm" / "experimental" / "verl" / "metrics.py"
_SPEC = importlib.util.spec_from_file_location("rllm_test_verl_metrics", _METRICS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_METRICS_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_METRICS_MODULE)
calculate_debug_metrics_compat = _METRICS_MODULE.calculate_debug_metrics_compat


class _DummyData:
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch


def test_calculate_debug_metrics_compat_prefers_response_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """An all-zero response_mask should short-circuit even if attention_mask is permissive."""
    monkeypatch.setattr(
        _METRICS_MODULE,
        "_load_verl_calculate_debug_metrics",
        lambda: pytest.fail("upstream helper should not be called for all-zero response_mask"),
    )

    data = _DummyData(
        {
            "rollout_log_probs": torch.zeros((1, 3), dtype=torch.float32),
            "old_log_probs": torch.zeros((1, 3), dtype=torch.float32),
            "response_mask": torch.zeros((1, 3), dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "responses": torch.ones((1, 3), dtype=torch.long),
        }
    )

    metrics = calculate_debug_metrics_compat(data)

    assert metrics["training/rollout_probs_diff_valid"] == 0
    assert math.isnan(metrics["training/rollout_probs_diff_max"])
    assert math.isnan(metrics["training/rollout_probs_diff_mean"])
    assert math.isnan(metrics["training/rollout_probs_diff_std"])
    assert math.isnan(metrics["training/rollout_actor_probs_pearson_corr"])


def test_calculate_debug_metrics_compat_uses_attention_mask_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When response_mask is absent, the wrapper should follow Verl's attention-mask fallback."""
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

    data = _DummyData(
        {
            "rollout_log_probs": torch.zeros((1, 3), dtype=torch.float32),
            "old_log_probs": torch.zeros((1, 3), dtype=torch.float32),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 1]], dtype=torch.long),
            "responses": torch.ones((1, 3), dtype=torch.long),
        }
    )

    metrics = calculate_debug_metrics_compat(data)

    assert calls == [data]
    assert metrics == expected_metrics


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

    data = _DummyData(
        {
            "rollout_log_probs": torch.zeros((1, 3), dtype=torch.float32),
            "old_log_probs": torch.zeros((1, 3), dtype=torch.float32),
            "response_mask": torch.tensor([[0, 1, 0]], dtype=torch.long),
            "responses": torch.ones((1, 3), dtype=torch.long),
        }
    )

    metrics = calculate_debug_metrics_compat(data)

    assert metrics["training/rollout_probs_diff_std"] == pytest.approx(0.0)
    assert math.isnan(metrics["training/rollout_actor_probs_pearson_corr"])


def test_calculate_debug_metrics_compat_short_circuits_empty_attention_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper should emulate newer upstream behavior for empty valid-token masks."""
    monkeypatch.setattr(
        _METRICS_MODULE,
        "_load_verl_calculate_debug_metrics",
        lambda: pytest.fail("upstream helper should not be called for all-zero effective mask"),
    )

    data = _DummyData(
        {
            "rollout_log_probs": torch.zeros((1, 3), dtype=torch.float32),
            "old_log_probs": torch.zeros((1, 3), dtype=torch.float32),
            "attention_mask": torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.long),
            "responses": torch.ones((1, 3), dtype=torch.long),
        }
    )

    metrics = calculate_debug_metrics_compat(data)

    assert metrics["training/rollout_probs_diff_valid"] == 0
    assert math.isnan(metrics["training/rollout_probs_diff_max"])
    assert math.isnan(metrics["training/rollout_actor_probs_pearson_corr"])


def test_calculate_debug_metrics_compat_short_circuits_empty_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty responses should also hit the compatibility fallback instead of delegating upstream."""
    monkeypatch.setattr(
        _METRICS_MODULE,
        "_load_verl_calculate_debug_metrics",
        lambda: pytest.fail("upstream helper should not be called for zero-length responses"),
    )

    data = _DummyData(
        {
            "rollout_log_probs": torch.zeros((1, 0), dtype=torch.float32),
            "old_log_probs": torch.zeros((1, 0), dtype=torch.float32),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "responses": torch.zeros((1, 0), dtype=torch.long),
        }
    )

    metrics = calculate_debug_metrics_compat(data)

    assert metrics["training/rollout_probs_diff_valid"] == 0
    assert math.isnan(metrics["training/rollout_probs_diff_std"])

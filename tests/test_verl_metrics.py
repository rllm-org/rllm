"""Tests for small Verl-specific metric helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_METRICS_PATH = Path(__file__).resolve().parents[1] / "rllm" / "experimental" / "verl" / "metrics.py"
_SPEC = importlib.util.spec_from_file_location("rllm_test_verl_metrics", _METRICS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_METRICS_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_METRICS_MODULE)
compute_rollout_probs_diff_metrics = _METRICS_MODULE.compute_rollout_probs_diff_metrics


def test_rollout_probs_diff_metrics_use_response_mask() -> None:
    """Internal zero-mask positions should not contribute to the drift metrics."""
    rollout_probs = torch.tensor([[0.2, 0.9, 0.5], [0.1, 0.8, 0.3]], dtype=torch.float32)
    actor_probs = torch.tensor([[0.4, 0.1, 0.2], [0.6, 0.7, 0.4]], dtype=torch.float32)
    response_mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.long)

    metrics = compute_rollout_probs_diff_metrics(
        rollout_log_probs=torch.log(rollout_probs),
        actor_log_probs=torch.log(actor_probs),
        response_mask=response_mask,
    )

    assert metrics["training/rollout_probs_diff_max"] == pytest.approx(0.3)
    assert metrics["training/rollout_probs_diff_mean"] == pytest.approx(0.2)
    assert metrics["training/rollout_probs_diff_std"] == pytest.approx(0.1)


def test_rollout_probs_diff_metrics_skip_empty_masks() -> None:
    """An all-zero response mask should produce no metrics instead of crashing."""
    response_mask = torch.zeros((2, 3), dtype=torch.long)

    metrics = compute_rollout_probs_diff_metrics(
        rollout_log_probs=torch.zeros((2, 3), dtype=torch.float32),
        actor_log_probs=torch.zeros((2, 3), dtype=torch.float32),
        response_mask=response_mask,
    )

    assert metrics == {}


def test_rollout_probs_diff_metrics_single_token_std_is_zero() -> None:
    """A single valid token should report zero std instead of NaN."""
    response_mask = torch.tensor([[0, 1, 0]], dtype=torch.long)

    metrics = compute_rollout_probs_diff_metrics(
        rollout_log_probs=torch.log(torch.tensor([[0.2, 0.6, 0.4]], dtype=torch.float32)),
        actor_log_probs=torch.log(torch.tensor([[0.1, 0.1, 0.3]], dtype=torch.float32)),
        response_mask=response_mask,
    )

    assert metrics["training/rollout_probs_diff_max"] == pytest.approx(0.5)
    assert metrics["training/rollout_probs_diff_mean"] == pytest.approx(0.5)
    assert metrics["training/rollout_probs_diff_std"] == pytest.approx(0.0)

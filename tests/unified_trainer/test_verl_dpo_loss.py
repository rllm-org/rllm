"""Pure tensor tests for the Verl DPO loss helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rllm.experimental.verl.dpo_loss import compute_dpo_loss, sequence_log_probs


def test_sequence_log_probs_sums_only_response_mask():
    log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
    mask = torch.tensor([[1, 0, 1]])

    assert sequence_log_probs(log_probs, mask).item() == pytest.approx(-4.0)


def test_compute_dpo_loss_for_ordered_chosen_rejected_pair():
    policy = torch.tensor([[-0.1, -0.1], [-1.0, -1.0]])
    ref = torch.zeros_like(policy)
    mask = torch.ones_like(policy, dtype=torch.bool)

    loss, metrics = compute_dpo_loss(policy, ref, mask, beta=1.0)

    expected_margin = torch.tensor(1.8)
    assert loss.item() == pytest.approx((-F.logsigmoid(expected_margin)).item())
    assert metrics["dpo/accuracy"].item() == pytest.approx(1.0)
    assert metrics["dpo/margin_mean"].item() == pytest.approx(1.8)
    assert metrics["dpo/chosen_logp_mean"].item() == pytest.approx(-0.2)
    assert metrics["dpo/rejected_logp_mean"].item() == pytest.approx(-2.0)


def test_compute_dpo_loss_honors_zero_weight_dummy_pairs():
    policy = torch.tensor(
        [
            [-0.1, -0.1],
            [-1.0, -1.0],
            [-5.0, -5.0],
            [-0.1, -0.1],
        ]
    )
    ref = torch.zeros_like(policy)
    mask = torch.ones_like(policy, dtype=torch.bool)

    loss, metrics = compute_dpo_loss(policy, ref, mask, beta=1.0, pair_weights=torch.tensor([1.0, 1.0, 0.0, 0.0]))

    expected_margin = torch.tensor(1.8)
    assert loss.item() == pytest.approx((-F.logsigmoid(expected_margin)).item())
    assert metrics["dpo/accuracy"].item() == pytest.approx(1.0)


def test_compute_dpo_loss_rejects_odd_row_count():
    with pytest.raises(ValueError, match="even number"):
        compute_dpo_loss(
            torch.zeros(3, 2),
            torch.zeros(3, 2),
            torch.ones(3, 2, dtype=torch.bool),
            beta=0.1,
        )


def test_compute_dpo_loss_rejects_non_adjacent_pair_order():
    with pytest.raises(ValueError, match="chosen/rejected"):
        compute_dpo_loss(
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.ones(2, 2, dtype=torch.bool),
            beta=0.1,
            is_chosen=torch.tensor([False, True]),
        )

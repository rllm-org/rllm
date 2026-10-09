import numpy as np
import pytest

from rllm.trainer.algorithms.advantage import calculate_reinforce_plus_plus_baseline_advantages


def _advantages(rewards):
    advantages, returns = calculate_reinforce_plus_plus_baseline_advantages(rewards, algorithm_config=None)
    assert all(np.array_equal(a, r) for a, r in zip(advantages, returns, strict=True))
    return advantages


def test_single_rollout_groups_keep_a_reward_signal():
    """With rollout.n == 1 every group has one reward; the baseline is 0, not the reward itself (#916)."""
    rewards = [np.array([1.0]), np.array([0.0]), np.array([0.5])]

    advantages = _advantages(rewards)

    flat = np.concatenate(advantages)
    assert not np.allclose(flat, 0.0)
    # Whitening keeps the order and sign of the rewards.
    assert flat[0] > flat[2] > flat[1]
    assert flat[1] == pytest.approx(0.0)


def test_multi_rollout_groups_use_the_group_mean_baseline():
    rewards = [np.array([1.0, 0.0]), np.array([0.5, 0.5])]

    advantages = _advantages(rewards)

    # Centered by the group mean, then whitened by the batch std.
    assert advantages[0][0] > 0 > advantages[0][1]
    assert advantages[0][0] == pytest.approx(-advantages[0][1])
    assert np.allclose(advantages[1], 0.0)


def test_mixed_group_sizes():
    rewards = [np.array([1.0, 0.0]), np.array([2.0])]

    advantages = _advantages(rewards)

    assert advantages[0][0] == pytest.approx(-advantages[0][1])
    assert advantages[1][0] > 0


def test_empty_input():
    assert calculate_reinforce_plus_plus_baseline_advantages([], algorithm_config=None) == ([], [])

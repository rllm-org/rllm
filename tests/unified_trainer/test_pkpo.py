"""Numerical tests for the PKPO advantage transform (arXiv:2505.15201).

Loads rl_algo.py directly (numpy-only) to avoid heavy transitive deps.
"""

import importlib.util
import os

import numpy as np

_RL_ALGO_PATH = os.path.join(os.path.dirname(__file__), "../../rllm/trainer/algorithms/rl_algo.py")
_spec = importlib.util.spec_from_file_location("rllm_rl_algo", _RL_ALGO_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
calculate_pkpo_advantages_per_group = _mod.calculate_pkpo_advantages_per_group


def test_k1_reduces_to_mean_centered_reward_over_n():
    """At k=1 the credit is reward/n, so advantages are (reward - mean) / n."""
    rewards = np.array([1.0, 2.0, 3.0, 4.0])
    adv, ret = calculate_pkpo_advantages_per_group(rewards, k=1)
    expected = (rewards - rewards.mean()) / len(rewards)
    assert np.allclose(adv, expected)
    assert np.allclose(ret, expected)


def test_binary_matches_theorem2():
    """n=4, c=2, k=2: correct credit = k/n = 0.5, incorrect = (k/n)*rho(3,2,1) = 1/3.

    Centering (mean credit = 5/12) gives advantages +/- 1/12.
    """
    rewards = np.array([1.0, 0.0, 1.0, 0.0])
    adv, _ = calculate_pkpo_advantages_per_group(rewards, k=2)
    expected = np.array([1 / 12, -1 / 12, 1 / 12, -1 / 12])
    assert np.allclose(adv, expected)


def test_advantage_is_non_decreasing_in_reward():
    """Advantage is weakly monotonic in reward (bottom samples can tie: the smallest
    is never the max of any k-subset for k>1)."""
    rewards = np.array([0.2, 0.9, 0.5, 0.1])
    adv, _ = calculate_pkpo_advantages_per_group(rewards, k=2)
    adv_by_reward = adv[np.argsort(rewards)]
    assert np.all(np.diff(adv_by_reward) >= -1e-9)


def test_advantages_are_centered():
    rewards = np.array([0.0, 0.3, 1.0, 0.7, 0.1])
    adv, _ = calculate_pkpo_advantages_per_group(rewards, k=3)
    assert abs(float(adv.sum())) < 1e-9


def test_uniform_rewards_give_zero_signal():
    rewards = np.array([1.0, 1.0, 1.0, 1.0])
    adv, _ = calculate_pkpo_advantages_per_group(rewards, k=2)
    assert np.allclose(adv, 0.0)


def test_k_clamped_to_n():
    """k > n behaves like k = n (all credits equal the max -> no signal after centering)."""
    rewards = np.array([1.0, 2.0, 3.0])
    adv_kn, _ = calculate_pkpo_advantages_per_group(rewards, k=3)
    adv_big, _ = calculate_pkpo_advantages_per_group(rewards, k=99)
    assert np.allclose(adv_kn, 0.0)
    assert np.allclose(adv_big, 0.0)


def test_single_and_empty_groups():
    for rewards in (np.array([]), np.array([2.5])):
        adv, ret = calculate_pkpo_advantages_per_group(rewards, k=2)
        assert adv.shape == rewards.shape
        assert np.allclose(adv, 0.0)

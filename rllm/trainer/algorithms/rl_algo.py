"""Algorithm for RL advantage estimation and loss functions"""

from math import comb

import numpy as np


def _comb(n: int, r: int) -> int:
    """Binomial coefficient, 0 outside the valid range (avoids math.comb ValueError)."""
    return comb(n, r) if n >= r >= 0 else 0


def calculate_grpo_advantages_per_group(rewards: np.ndarray, norm_adv_by_std_in_grpo=True, episilon=1e-6) -> tuple[np.ndarray, np.ndarray]:
    if len(rewards) <= 1:
        group_mean, group_std = 0.0, 1.0
    else:
        group_mean = np.mean(rewards)
        group_std = np.std(rewards)

    if norm_adv_by_std_in_grpo:
        advantages = (rewards - group_mean) / (group_std + episilon)
    else:
        advantages = rewards - group_mean

    return advantages, advantages


def calculate_rloo_advantages_per_group(rewards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_trajs = len(rewards)
    if num_trajs <= 1:
        return rewards, rewards

    advantages = num_trajs / (num_trajs - 1) * (rewards - rewards.mean())
    return advantages, advantages


def calculate_pkpo_advantages_per_group(rewards: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """PKPO reward transform for pass@k / max@k optimization (arXiv:2505.15201).

    Credits each sample with ``(1/C(n,k)) * sum, over k-subsets containing it, of the
    subset's max reward`` -- an unbiased estimator of the pass@k gradient -- then centers
    by the group mean. Tied rewards share credit equally. ``k`` is clamped to ``[1, n]``;
    use ``k < n`` (at ``k == n`` every credit equals the group max, so centering leaves no
    signal). Reduces to mean-centered reward at ``k == 1``.
    """
    rewards = np.asarray(rewards, dtype=float)
    n = len(rewards)
    if n <= 1:
        return np.zeros(n), np.zeros(n)
    k = max(1, min(k, n))

    order = np.argsort(rewards, kind="stable")
    g = rewards[order]  # ascending
    cnk = comb(n, k)

    # Credit for the p-th smallest sample (0-indexed): it is the subset max for the
    # C(p, k-1) subsets drawn from smaller samples, and a non-max member otherwise.
    credit_sorted = np.empty(n, dtype=float)
    suffix = 0.0  # running sum_{q>p} C(q-1, k-2) * g[q]
    for p in range(n - 1, -1, -1):
        credit_sorted[p] = (_comb(p, k - 1) * g[p] + suffix) / cnk
        suffix += _comb(p - 1, k - 2) * g[p]

    credit = np.empty(n, dtype=float)
    credit[order] = credit_sorted
    for val in np.unique(rewards):  # exchangeable samples get equal credit
        mask = rewards == val
        credit[mask] = credit[mask].mean()

    advantages = credit - credit.mean()
    return advantages, advantages

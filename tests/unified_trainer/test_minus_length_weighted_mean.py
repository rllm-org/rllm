"""Tests for the minus_length_weighted_mean advantage estimator.

Baseline = length-weighted mean of shaped rewards over the whole batch pool;
in token mode (steps carrying metadata["per_token_rewards"]) the baseline is the
mean per-token reward over all tokens and advantages are emitted per token.
The two doc examples are reproduced exactly.
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

from rllm.types import Step, Trajectory, TrajectoryGroup
from rllm.trainer.algorithms.advantage import (
    collect_reward_and_advantage_from_trajectory_groups,
    get_adv_estimator,
)
from rllm.trainer.algorithms.config import AlgorithmConfig, rLLMAdvantageEstimator


def _step(per_token_rewards, n):
    return Step(
        response_ids=list(range(n)),
        metadata={"per_token_rewards": per_token_rewards} if per_token_rewards is not None else None,
    )


def _traj(reward, steps):
    t = Trajectory(steps=steps)
    t.reward = reward
    return t


def _approx(xs):
    return [pytest.approx(x, abs=1e-4) for x in xs]


class TestMinusLengthWeightedMean:
    def _fn(self):
        return get_adv_estimator(rLLMAdvantageEstimator.MINUS_LENGTH_WEIGHTED_MEAN)

    def test_token_mode_matches_doc_example(self):
        # Doc example 2: baseline = (1+1+0.8+0.6+0+0)/6 = 0.5667
        a = _traj(1.0, [_step([1.0, 1.0, 0.8, 0.6], 4)])
        b = _traj(0.0, [_step([0.0, 0.0], 2)])
        g = TrajectoryGroup(group_id="g", trajectories=[a, b])
        adv, ret = self._fn()(rewards=[np.array([1.0, 0.0])], algorithm_config=None, traj_groups=[g])

        assert a.steps[0].advantage == _approx([0.4333, 0.4333, 0.2333, 0.0333])
        assert b.steps[0].advantage == _approx([-0.5667, -0.5667])
        # returned scalars are the sample advantages (shaped_reward - baseline)
        assert list(adv[0]) == _approx([0.4333, -0.5667])
        assert list(ret[0]) == _approx([0.4333, -0.5667])

    def test_outcome_mode_matches_doc_example(self):
        # Doc example 1: no per-token rewards -> length-weighted mean = 0.6
        a = _traj(0.9, [_step(None, 100)])
        b = _traj(0.0, [_step(None, 50)])
        g = TrajectoryGroup(group_id="g", trajectories=[a, b])
        adv, _ = self._fn()(rewards=[np.array([0.9, 0.0])], algorithm_config=None, traj_groups=[g])

        assert list(adv[0]) == _approx([0.3, -0.6])
        # a constant per-token advantage is broadcast across the response
        assert a.steps[0].advantage == _approx([0.3] * 100)
        assert b.steps[0].advantage == _approx([-0.6] * 50)

    def test_mixed_batch_shares_one_pooled_baseline(self):
        # token-mode + outcome-mode trajectories pool into a single token mean
        a = _traj(1.0, [_step([1.0, 1.0, 0.8, 0.6], 4)])
        c = _traj(0.0, [_step(None, 2)])  # contributes 0.0 at each of its 2 tokens
        g = TrajectoryGroup(group_id="g", trajectories=[a, c])
        self._fn()(rewards=[np.array([1.0, 0.0])], algorithm_config=None, traj_groups=[g])

        baseline = (1.0 + 1.0 + 0.8 + 0.6 + 0.0 + 0.0) / 6
        assert a.steps[0].advantage == _approx([1.0 - baseline, 1.0 - baseline, 0.8 - baseline, 0.6 - baseline])
        assert c.steps[0].advantage == _approx([-baseline, -baseline])

    def test_multi_step_trajectory_pools_across_steps(self):
        d = _traj(0.8, [_step([0.5, 0.5], 2), _step([1.0], 1)])
        g = TrajectoryGroup(group_id="g", trajectories=[d])
        self._fn()(rewards=[np.array([0.8])], algorithm_config=None, traj_groups=[g])

        baseline = (0.5 + 0.5 + 1.0) / 3
        assert d.steps[0].advantage == _approx([0.5 - baseline, 0.5 - baseline])
        assert d.steps[1].advantage == _approx([1.0 - baseline])

    def test_length_mismatch_falls_back_to_trajectory_reward(self):
        # per_token_rewards shorter than response_ids -> that step uses the scalar reward
        a = _traj(0.5, [_step([1.0], 3)])  # 1 ptr vs 3 tokens
        g = TrajectoryGroup(group_id="g", trajectories=[a])
        self._fn()(rewards=[np.array([0.5])], algorithm_config=None, traj_groups=[g])
        # pooled over 3 tokens each contributing the trajectory reward 0.5 -> baseline 0.5
        assert a.steps[0].advantage == _approx([0.0, 0.0, 0.0])

    def test_end_to_end_via_collect_writes_per_token_and_metrics(self):
        alg = AlgorithmConfig.from_config(OmegaConf.create({"adv_estimator": "minus_length_weighted_mean"}))
        assert alg.estimator == rLLMAdvantageEstimator.MINUS_LENGTH_WEIGHTED_MEAN

        a = _traj(1.0, [_step([1.0, 1.0, 0.8, 0.6], 4)])
        b = _traj(0.0, [_step([0.0, 0.0], 2)])
        g = TrajectoryGroup(group_id="g", group_role="default", trajectories=[a, b])
        metrics = collect_reward_and_advantage_from_trajectory_groups([g], alg)

        # per-token lists survive the caller (not collapsed to scalar broadcast)
        assert isinstance(a.steps[0].advantage, list) and len(a.steps[0].advantage) == 4
        assert a.steps[0].advantage == _approx([0.4333, 0.4333, 0.2333, 0.0333])
        # scalar sample advantages still feed advantage/* metrics
        assert any(k.startswith("advantage/") and k.endswith("/mean") for k in metrics)

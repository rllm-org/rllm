"""Tests for per-rollout advantage dedup (subagent lineages).

When a rollout contributes multiple trajectories to a GRPO group — the parent
agent plus each subagent lineage, all sharing the rollout's single reward — the
baseline must count that rollout ONCE. These tests pin:

1. ``_dedup_rollout_rewards`` collapses by ``rollout_idx``.
2. Dedup is a no-op for standard single-trajectory-per-rollout groups.
3. For a multi-lineage rollout the baseline is over per-rollout rewards, and
   every lineage of a rollout receives that rollout's advantage.
"""

import numpy as np
import pytest

from rllm.trainer.algorithms.advantage import (
    _dedup_rollout_rewards,
    collect_reward_and_advantage_from_trajectory_groups,
)
from rllm.trainer.algorithms.config import AlgorithmConfig
from rllm.types import Step, Trajectory, TrajectoryGroup


def _traj(reward: float) -> Trajectory:
    return Trajectory(steps=[Step(response_ids=[1])], reward=reward)


def _group(rewards_and_rollouts: list[tuple[float, int]], group_id: str = "task:default_0") -> TrajectoryGroup:
    trajs = [_traj(r) for r, _ in rewards_and_rollouts]
    meta = [{"rollout_idx": ridx} for _, ridx in rewards_and_rollouts]
    return TrajectoryGroup(trajectories=trajs, group_id=group_id, metadata=meta)


def _step_adv(traj: Trajectory) -> float:
    return traj.steps[0].advantage


class TestDedupRolloutRewards:
    def test_collapses_by_rollout_idx(self):
        group = _group([(1.0, 0), (1.0, 0), (1.0, 0), (0.0, 1)])
        rollout_rewards, traj_to_pos = _dedup_rollout_rewards(group)
        assert rollout_rewards.tolist() == [1.0, 0.0]  # one per rollout
        assert traj_to_pos == [0, 0, 0, 1]  # 3 lineages of rollout 0, then rollout 1

    def test_no_metadata_is_no_dedup(self):
        # Misaligned/empty metadata → each trajectory is its own rollout.
        trajs = [_traj(1.0), _traj(0.0)]
        group = TrajectoryGroup(trajectories=trajs, group_id="task:default_0", metadata=[])
        rollout_rewards, traj_to_pos = _dedup_rollout_rewards(group)
        assert rollout_rewards.tolist() == [1.0, 0.0]
        assert traj_to_pos == [0, 1]


class TestCollectDedup:
    def test_dedup_noop_for_single_trajectory_rollouts(self):
        # Two rollouts, one trajectory each → dedup changes nothing.
        group = _group([(1.0, 0), (0.0, 1)])
        collect_reward_and_advantage_from_trajectory_groups([group], AlgorithmConfig())
        # GRPO over [1, 0]: mean 0.5, std 0.5 → ±1.0
        assert _step_adv(group.trajectories[0]) == pytest.approx(1.0, abs=1e-4)
        assert _step_adv(group.trajectories[1]) == pytest.approx(-1.0, abs=1e-4)

    def test_multilineage_rollout_baselined_once(self):
        # Rollout 0 spawned 3 lineages (reward 1.0), rollout 1 has 1 (reward 0.0).
        # Baseline must be over per-rollout rewards [1, 0] (mean 0.5, std 0.5),
        # NOT the per-trajectory [1,1,1,0] (which would give mean 0.75).
        group = _group([(1.0, 0), (1.0, 0), (1.0, 0), (0.0, 1)])
        collect_reward_and_advantage_from_trajectory_groups([group], AlgorithmConfig())
        # Every lineage of rollout 0 gets the same advantage = (1-0.5)/0.5 = 1.0
        for i in range(3):
            assert _step_adv(group.trajectories[i]) == pytest.approx(1.0, abs=1e-4)
        # Rollout 1 gets (0-0.5)/0.5 = -1.0
        assert _step_adv(group.trajectories[3]) == pytest.approx(-1.0, abs=1e-4)

        # Contrast: the per-trajectory (non-deduped) baseline would differ.
        naive_mean = np.mean([1, 1, 1, 0])
        assert naive_mean != pytest.approx(0.5)  # 0.75 — the skew dedup removes

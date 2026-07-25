"""Tests for the per-role {all, effective} reward matrix in TrajectoryGroupBuffer.

`reward/{role}/all/*` (every trajectory that ran, pre-filter) is the honest
per-role performance view; `reward/{role}/effective/*` (post every filter) is
what actually trains. Grouping by trajectory role means multi-agent runs
(e.g. solver/judge) report each role separately, matching reward/{role}/*.
See buffer._record_reward_by_role / _record_reward_stats.
"""

from rllm.agents.agent import Step, Trajectory
from rllm.trainer.buffer import TrajectoryGroupBuffer
from rllm.trainer.metrics_aggregator import MetricsAggregator


def _buffer_with_aggregator() -> TrajectoryGroupBuffer:
    """A buffer whose only wired-up dependency is the aggregator. The metric
    helpers under test read `self._aggregator` and nothing else, so bypass
    __init__ (which builds asyncio primitives / offload dirs)."""
    buf = TrajectoryGroupBuffer.__new__(TrajectoryGroupBuffer)
    buf._aggregator = MetricsAggregator()
    return buf


def test_record_reward_stats_empty_is_noop():
    # A fully-filtered task contributes no effective rewards -> it drops out of
    # reward/{role}/effective/* (the intended {all, effective} asymmetry).
    buf = _buffer_with_aggregator()
    buf._record_reward_stats("reward/opencode/effective", [])
    assert buf._aggregator.flush() == {}


def test_record_reward_stats_values():
    buf = _buffer_with_aggregator()
    buf._record_reward_stats("reward/opencode/all", [0.0, 1.0, 1.0])
    m = buf._aggregator.flush()
    assert m["reward/opencode/all/mean"] == 2.0 / 3.0
    assert m["reward/opencode/all/max"] == 1.0
    assert m["reward/opencode/all/min"] == 0.0
    assert m["reward/opencode/all/std"] > 0.0


def test_reward_by_role_reports_each_role_separately():
    # Multi-agent: solver + judge each get their own reward/<role>/all/*.
    buf = _buffer_with_aggregator()
    trajs = [
        Trajectory(name="solver", steps=[], reward=1.0),
        Trajectory(name="solver", steps=[], reward=0.0),
        Trajectory(name="judge", steps=[], reward=0.5),
        Trajectory(name="judge", steps=[], reward=0.5),
    ]
    buf._record_reward_by_role("all", trajs, with_difficulty=True)
    m = buf._aggregator.flush()

    assert m["reward/solver/all/mean"] == 0.5
    assert m["reward/solver/all/max"] == 1.0
    assert m["reward/judge/all/mean"] == 0.5
    # solver: 1 solved / 1 failed -> mixed (carries GRPO signal)
    assert m["reward/solver/solved_some"] == 1.0
    assert m["reward/solver/solved_none"] == 0.0
    assert m["reward/solver/solved_all"] == 0.0
    # judge: both > 0 -> uniform "all solved"
    assert m["reward/judge/solved_all"] == 1.0
    assert m["reward/judge/solved_some"] == 0.0


def test_reward_by_role_effective_has_no_difficulty():
    # Difficulty (solved_*) is only meaningful pre-filter, so it's recorded on
    # the `all` subset; effective is post-filter (uniform groups already gone).
    buf = _buffer_with_aggregator()
    trajs = [Trajectory(name="opencode", steps=[], reward=1.0), Trajectory(name="opencode", steps=[], reward=0.0)]
    buf._record_reward_by_role("effective", trajs)
    m = buf._aggregator.flush()
    assert m["reward/opencode/effective/mean"] == 0.5
    assert not any(k.startswith("reward/opencode/solved_") for k in m)


def test_reward_by_role_solved_none_when_role_all_fail():
    buf = _buffer_with_aggregator()
    trajs = [Trajectory(name="opencode", steps=[], reward=0.0) for _ in range(3)]
    buf._record_reward_by_role("all", trajs, with_difficulty=True)
    m = buf._aggregator.flush()
    assert m["reward/opencode/solved_none"] == 1.0
    assert m["reward/opencode/solved_some"] == 0.0
    assert m["reward/opencode/solved_all"] == 0.0


def test_reward_by_role_falls_back_to_last_step_reward():
    buf = _buffer_with_aggregator()
    trajs = [Trajectory(name="opencode", steps=[Step(reward=0.4)], reward=None)]
    buf._record_reward_by_role("all", trajs)
    m = buf._aggregator.flush()
    assert m["reward/opencode/all/mean"] == 0.4


def test_reward_by_role_empty_is_noop():
    buf = _buffer_with_aggregator()
    buf._record_reward_by_role("effective", [])
    assert buf._aggregator.flush() == {}

"""Tests for the {all, effective} reward matrix in TrajectoryGroupBuffer.

`reward/all/*` (recorded over every rollout, errors counted as 0) is a true
reflection of performance; `reward/effective/*` (post every filter) is what
actually trains. See buffer._record_episode_metrics / _record_reward_stats.
"""

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.trainer.buffer import TrajectoryGroupBuffer
from rllm.trainer.metrics_aggregator import MetricsAggregator
from rllm.workflows.workflow import TerminationReason


def _buffer_with_aggregator() -> TrajectoryGroupBuffer:
    """A buffer whose only wired-up dependency is the aggregator. The metric
    helpers under test read `self._aggregator` and nothing else, so bypass
    __init__ (which builds asyncio primitives / offload dirs)."""
    buf = TrajectoryGroupBuffer.__new__(TrajectoryGroupBuffer)
    buf._aggregator = MetricsAggregator()
    return buf


def test_episode_scalar_reward_uses_trajectory_reward():
    ep = Episode(id="t:0", trajectories=[Trajectory(steps=[], reward=0.7)], is_correct=True)
    assert TrajectoryGroupBuffer._episode_scalar_reward(ep) == 0.7


def test_episode_scalar_reward_errored_episode_is_zero():
    # No scored trajectory (the shape agentflow_engine emits on ERROR) -> 0.0,
    # NOT dropped. This is the crux: compact-filtering removes these before
    # reward/* is computed, inflating it; reward/all keeps them as failures.
    ep = Episode(id="t:1", trajectories=[], termination_reason=TerminationReason.ERROR, is_correct=False)
    assert TrajectoryGroupBuffer._episode_scalar_reward(ep) == 0.0


def test_episode_scalar_reward_falls_back_to_last_step():
    ep = Episode(id="t:2", trajectories=[Trajectory(steps=[Step(reward=0.4)], reward=None)])
    assert TrajectoryGroupBuffer._episode_scalar_reward(ep) == 0.4


def test_record_reward_stats_empty_is_noop():
    # A fully-filtered task contributes no effective rewards -> it drops out of
    # reward/effective/* entirely (the intended {all, effective} asymmetry).
    buf = _buffer_with_aggregator()
    buf._record_reward_stats("reward/effective", [])
    assert buf._aggregator.flush() == {}


def test_record_reward_stats_values():
    buf = _buffer_with_aggregator()
    buf._record_reward_stats("reward/all", [0.0, 1.0, 1.0])
    m = buf._aggregator.flush()
    assert m["reward/all/mean"] == 2.0 / 3.0
    assert m["reward/all/max"] == 1.0
    assert m["reward/all/min"] == 0.0
    assert m["reward/all/std"] > 0.0


def test_all_view_counts_errored_as_zero_and_decomposes_difficulty():
    # 4 rollouts of one prompt group: 1 solved, 2 genuinely failed, 1 errored.
    episodes = [
        Episode(id="t:0", trajectories=[Trajectory(steps=[], reward=1.0)], is_correct=True),
        Episode(id="t:1", trajectories=[Trajectory(steps=[], reward=0.0)], is_correct=False),
        Episode(id="t:2", trajectories=[Trajectory(steps=[], reward=0.0)], is_correct=False),
        Episode(id="t:3", trajectories=[], termination_reason=TerminationReason.ERROR, is_correct=False),
    ]
    buf = _buffer_with_aggregator()
    buf._record_episode_metrics(episodes)
    m = buf._aggregator.flush()

    # Errored rollout scored 0 (kept, not dropped): mean = (1+0+0+0)/4 = 0.25.
    assert m["reward/all/mean"] == 0.25
    assert m["reward/all/max"] == 1.0
    assert m["reward/all/min"] == 0.0
    # Mixed group -> carries GRPO signal.
    assert m["reward/all/solved_some"] == 1.0
    assert m["reward/all/solved_none"] == 0.0
    assert m["reward/all/solved_all"] == 0.0
    # Sanity: the pre-existing all-episodes accuracy agrees (1 of 4 correct).
    assert m["episode/correct"] == 0.25


def test_all_view_solved_none_when_every_rollout_fails():
    episodes = [
        Episode(id="t:0", trajectories=[Trajectory(steps=[], reward=0.0)], is_correct=False),
        Episode(id="t:1", trajectories=[], termination_reason=TerminationReason.ERROR, is_correct=False),
    ]
    buf = _buffer_with_aggregator()
    buf._record_episode_metrics(episodes)
    m = buf._aggregator.flush()
    assert m["reward/all/solved_none"] == 1.0
    assert m["reward/all/solved_some"] == 0.0
    assert m["reward/all/solved_all"] == 0.0

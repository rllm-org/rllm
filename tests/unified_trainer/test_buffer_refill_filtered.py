"""Tests for rejection_sample.refill_filtered_uniform_groups in TrajectoryGroupBuffer.

When a uniform (zero-advantage) group is dropped: True (default) refills it (backfill);
False queues an empty placeholder that counts toward the step but trains nothing. Scoped to
uniform drops only -- min-trajs / compact-filtering drops always refill. reward/{role}/all
counts trajectories that ran; reward/{role}/effective only those trained on.
"""

import pytest

from rllm.agents.agent import Step, Trajectory
from rllm.trainer.algorithms import (
    AlgorithmConfig,
    CompactFilteringConfig,
    RejectionSamplingConfig,
    TransformConfig,
)
from rllm.trainer.buffer import TaskBatch, TrajectoryGroupBuffer
from rllm.trainer.metrics_aggregator import MetricsAggregator
from rllm.trainer.sync_coordinator import SyncCoordinator, SyncCoordinatorConfig
from rllm.types import Episode, TerminationReason

GROUP_SIZE = 2


def _episode(reward: float, idx: int) -> Episode:
    step = Step(prompt_ids=[1, 2, 3], response_ids=[4, 5], logprobs=[0.0, 0.0], reward=reward)
    traj = Trajectory(name="agent", steps=[step], reward=reward)
    return Episode(id=f"task0:{idx}", task={"q": "x"}, termination_reason=TerminationReason.ENV_DONE, is_correct=reward > 0, trajectories=[traj])


def _make_buffer(rs_config: RejectionSamplingConfig, *, min_trajs_group_size: int = GROUP_SIZE) -> tuple[TrajectoryGroupBuffer, SyncCoordinator, MetricsAggregator]:
    coordinator = SyncCoordinator(SyncCoordinatorConfig(mini_batch_size=2, group_size=GROUP_SIZE, staleness_threshold=0.0, trigger_parameter_sync_step=1, max_concurrent_rollouts=8))
    aggregator = MetricsAggregator()
    buffer = TrajectoryGroupBuffer(
        group_size=min_trajs_group_size,
        coordinator=coordinator,
        aggregator=aggregator,
        algorithm_config=AlgorithmConfig(),
        transform_config=TransformConfig(),
        cf_config=CompactFilteringConfig(),
        rs_config=rs_config,
    )
    return buffer, coordinator, aggregator


async def _feed_group(buffer: TrajectoryGroupBuffer, coordinator: SyncCoordinator, rewards: list[float]) -> None:
    """Simulate one dispatched task whose group_size rollouts all arrive."""
    coordinator.on_group_dispatched()  # in_flight -> 1
    for i, r in enumerate(rewards):
        await buffer.add_episode("task0", _episode(r, i))


@pytest.mark.asyncio
async def test_uniform_group_refill_true_backfills():
    rs = RejectionSamplingConfig(mode="group", min_trajs_per_group=2, filter_uniform_groups=True, refill_filtered_uniform_groups=True)
    buffer, coordinator, aggregator = _make_buffer(rs)

    await _feed_group(buffer, coordinator, [1.0, 1.0])  # uniform -> advantage 0 -> dropped

    # Nothing queued; the slot was freed so generation backfills a replacement.
    assert buffer._queue.qsize() == 0
    assert buffer._training_queue_size == 0
    assert buffer._filtered_count == 1
    assert coordinator.stats()["async/in_flight_groups"] == 0  # on_group_filtered freed it

    m = aggregator.flush()
    assert any(k.startswith("reward/agent/all/") for k in m)  # it ran
    assert not any(k.startswith("reward/agent/effective/") for k in m)  # but wasn't trained on


@pytest.mark.asyncio
async def test_uniform_group_refill_false_counts_toward_step():
    rs = RejectionSamplingConfig(mode="group", min_trajs_per_group=2, filter_uniform_groups=True, refill_filtered_uniform_groups=False)
    buffer, coordinator, aggregator = _make_buffer(rs)

    await _feed_group(buffer, coordinator, [1.0, 1.0])  # uniform -> dropped, but counted

    # An empty placeholder occupies a step slot; the slot is NOT backfilled.
    assert buffer._queue.qsize() == 1
    assert buffer._training_queue_size == 1
    assert buffer._filtered_count == 1
    assert coordinator.stats()["async/in_flight_groups"] == 1  # on_group_filtered NOT called

    item = buffer._queue.get_nowait()
    assert isinstance(item, TaskBatch)
    assert item.groups == []  # trains nothing

    m = aggregator.flush()
    assert any(k.startswith("reward/agent/all/") for k in m)
    assert not any(k.startswith("reward/agent/effective/") for k in m)


@pytest.mark.asyncio
async def test_min_trajs_drop_always_refills_even_in_count_mode():
    # Scoping guarantee: a min-trajs drop (missing data, NOT a uniform drop) always refills,
    # even with refill_filtered_uniform_groups=False. group has 2 trajs < min_trajs_per_group=3.
    rs = RejectionSamplingConfig(mode="group", min_trajs_per_group=3, filter_uniform_groups=True, refill_filtered_uniform_groups=False)
    buffer, coordinator, aggregator = _make_buffer(rs)

    await _feed_group(buffer, coordinator, [1.0, 0.0])

    assert buffer._queue.qsize() == 0  # no placeholder -- refilled, not counted
    assert buffer._training_queue_size == 0
    assert buffer._filtered_count == 1
    assert coordinator.stats()["async/in_flight_groups"] == 0  # slot freed for backfill


@pytest.mark.asyncio
async def test_nonuniform_group_queued_normally():
    # Positive control: a group with real signal is queued with its groups regardless of the
    # refill flag, and contributes to reward/*/effective.
    rs = RejectionSamplingConfig(mode="group", min_trajs_per_group=2, filter_uniform_groups=True, refill_filtered_uniform_groups=False)
    buffer, coordinator, aggregator = _make_buffer(rs)

    await _feed_group(buffer, coordinator, [1.0, 0.0])  # mixed -> nonzero advantage -> survives

    assert buffer._queue.qsize() == 1
    assert buffer._filtered_count == 0
    item = buffer._queue.get_nowait()
    assert len(item.groups) == 1
    assert len(item.groups[0].trajectories) == 2

    m = aggregator.flush()
    assert any(k.startswith("reward/agent/all/") for k in m)
    assert any(k.startswith("reward/agent/effective/") for k in m)

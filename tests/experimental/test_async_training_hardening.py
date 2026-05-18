import asyncio

import pytest

from rllm.experimental.buffer import TrajectoryGroupBuffer
from rllm.experimental.common.config import (
    AlgorithmConfig,
    CompactFilteringConfig,
    RejectionSamplingConfig,
    TransformConfig,
)
from rllm.experimental.common.transform import transform_episodes_to_trajectory_groups
from rllm.experimental.metrics import MetricsAggregator
from rllm.experimental.sync_coordinator import SyncCoordinator, SyncCoordinatorConfig
from rllm.types import Episode
from rllm.workflows.workflow import TerminationReason


def test_transform_metrics_handle_all_filtered_groups():
    episodes = [Episode(id=f"task:{i}", trajectories=[], termination_reason=TerminationReason.TIMEOUT) for i in range(2)]
    cf_config = CompactFilteringConfig(enable=True, mask_timeout=True)

    groups, metrics = transform_episodes_to_trajectory_groups(
        episodes,
        TransformConfig(),
        cf_config,
    )

    assert groups == []
    assert metrics["groups/num_groups"] == 0
    assert metrics["groups/num_trajs_after_filter"] == 0
    assert metrics["groups/avg_group_size"] == 0.0
    assert metrics["groups/max_group_size"] == 0
    assert metrics["groups/min_group_size"] == 0


@pytest.mark.asyncio
async def test_buffer_all_filtered_group_decrements_in_flight_without_queueing_batch():
    coordinator = SyncCoordinator(
        SyncCoordinatorConfig(
            mini_batch_size=1,
            group_size=2,
            staleness_threshold=0.0,
            trigger_parameter_sync_step=1,
        )
    )
    aggregator = MetricsAggregator()
    buffer = TrajectoryGroupBuffer(
        group_size=2,
        coordinator=coordinator,
        aggregator=aggregator,
        algorithm_config=AlgorithmConfig(),
        transform_config=TransformConfig(),
        cf_config=CompactFilteringConfig(enable=True, mask_timeout=True),
        rs_config=RejectionSamplingConfig(),
    )

    coordinator.on_group_dispatched()
    assert (
        await buffer.add_episode(
            "task",
            Episode(id="task:0", trajectories=[], termination_reason=TerminationReason.TIMEOUT),
        )
        is False
    )
    assert (
        await buffer.add_episode(
            "task",
            Episode(id="task:1", trajectories=[], termination_reason=TerminationReason.TIMEOUT),
        )
        is True
    )

    assert coordinator.stats()["async/in_flight_groups"] == 0
    assert coordinator.stats()["async/quota_used"] == 0
    assert buffer.stats()["async/buffer_qsize"] == 0
    assert buffer.stats()["async/buffer_filtered"] == 1

    metrics = aggregator.flush()
    assert metrics["groups/num_groups"] == 0
    assert metrics["groups/dropped_min_trajs"] == 0

    buffer.mark_generation_complete()
    assert await buffer.get_many(1) is None


@pytest.mark.asyncio
async def test_sync_coordinator_surfaces_background_task_errors():
    coordinator = SyncCoordinator(
        SyncCoordinatorConfig(
            mini_batch_size=1,
            group_size=1,
            staleness_threshold=0.0,
            trigger_parameter_sync_step=1,
        )
    )

    async def fail():
        raise ValueError("boom")

    task = asyncio.create_task(fail())
    coordinator.track_task(task)
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="Async rollout task failed") as exc_info:
        await coordinator.wait_for_drain()

    assert isinstance(exc_info.value.__cause__, ValueError)

import asyncio
import threading

import pytest

from rllm.trainer.algorithms import AlgorithmConfig, CompactFilteringConfig, RejectionSamplingConfig, TransformConfig
from rllm.trainer.buffer import TrajectoryGroupBuffer
from rllm.trainer.metrics_aggregator import MetricsAggregator
from rllm.trainer.sync_coordinator import SyncCoordinator, SyncCoordinatorConfig
from rllm.types import Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationReason


@pytest.fixture
def buffer(tmp_path):
    return TrajectoryGroupBuffer(
        group_size=2,
        coordinator=SyncCoordinator(SyncCoordinatorConfig(mini_batch_size=1, group_size=2, staleness_threshold=0, trigger_parameter_sync_step=1)),
        aggregator=MetricsAggregator(),
        algorithm_config=AlgorithmConfig(),
        transform_config=TransformConfig(),
        cf_config=CompactFilteringConfig(),
        rs_config=RejectionSamplingConfig(),
        episode_offload_dir=str(tmp_path),
    )


def episode(index):
    return Episode(
        id=f"same-task:{index}",
        termination_reason=TerminationReason.ENV_DONE,
        trajectories=[Trajectory(name="agent", reward=index, steps=[Step(prompt_ids=[1, 2], response_ids=[10 + index], logprobs=[-0.5])])],
    )


@pytest.mark.asyncio
async def test_concurrent_offloads_preserve_both_episodes(buffer, tmp_path, monkeypatch):
    writers = threading.Barrier(2)
    dump = buffer._pickle_dump

    def concurrent_dump(path, value):
        writers.wait(timeout=5)
        dump(path, value)

    monkeypatch.setattr(buffer, "_pickle_dump", concurrent_dump)
    await asyncio.gather(*(buffer.add_episode("same-task", episode(index)) for index in range(2)))
    buffer.mark_generation_complete()
    batch = await asyncio.wait_for(buffer.get(), timeout=5)
    assert {ep.id for ep in batch.episodes} == {"same-task:0", "same-task:1"}
    assert {(traj.reward, tuple(traj.steps[0].prompt_ids), tuple(traj.steps[0].response_ids)) for group in batch.groups for traj in group.trajectories} == {
        (0, (1, 2), (10,)),
        (1, (1, 2), (11,)),
    }
    assert await buffer.get() is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_cancelled_offload_settles_write_and_removes_its_file(buffer, tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    sentinel = tmp_path / "user-owned.pkl"
    sentinel.write_text("keep")
    dump = buffer._pickle_dump

    def blocked_dump(path, value):
        started.set()
        try:
            if not release.wait(timeout=5):
                raise TimeoutError("write was not released")
            dump(path, value)
        finally:
            finished.set()

    monkeypatch.setattr(buffer, "_pickle_dump", blocked_dump)
    task = asyncio.create_task(buffer.add_episode("task", episode(0)))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "Cancellation returned while its disk writer was still running"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert finished.is_set()
        assert list(tmp_path.iterdir()) == [sentinel]
        assert sentinel.read_text() == "keep"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 5)

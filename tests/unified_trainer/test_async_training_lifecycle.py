import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from rllm.data.dataloader import StatefulTaskDataLoader
from rllm.trainer.algorithms import AlgorithmConfig, CompactFilteringConfig, RejectionSamplingConfig, TransformConfig
from rllm.trainer.algorithms.config import AsyncTrainingConfig
from rllm.trainer.unified_trainer import UnifiedTrainer
from rllm.types import Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationReason


@pytest.fixture
def make_trainer(monkeypatch):
    monkeypatch.setattr("rllm.trainer.unified_trainer.print_config_table", Mock())

    def make(dataset, rollout):
        trainer = object.__new__(UnifiedTrainer)
        trainer.config = OmegaConf.create({"rllm": {"rollout": {"n": 2}, "trainer": {"total_epochs": 1, "total_batches": -1, "val_before_train": False}}})
        trainer.rllm_config = trainer.config.rllm
        trainer.async_config = AsyncTrainingConfig(enable=True, staleness_threshold=1.0)
        trainer.algorithm_config = AlgorithmConfig()
        trainer.transform_config = TransformConfig()
        trainer.cf_config = CompactFilteringConfig()
        trainer.rs_config = RejectionSamplingConfig()
        trainer._train_dataloader = StatefulTaskDataLoader(dataset, 1, shuffle=False)
        trainer._total_training_steps = len(dataset)
        trainer._gateway = trainer._remote_runtime = None
        trainer.agent_workflow_engine = SimpleNamespace(raise_on_error=False, set_training_step=Mock(), process_task_with_retry=rollout)
        trainer.backend = SimpleNamespace(on_train_start=AsyncMock(), on_train_end=AsyncMock(), on_epoch_start=AsyncMock(), on_epoch_end=AsyncMock())
        return trainer

    return make


@asynccontextmanager
async def assert_no_leaked_tasks():
    before = asyncio.all_tasks()
    try:
        yield
        assert asyncio.all_tasks() == before
    finally:
        # Clean up leaks even on the unfixed baseline so tests remain isolated.
        remaining = asyncio.all_tasks() - before
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)


@pytest.mark.asyncio
async def test_dataset_failure_reaches_fit_caller(make_trainer):
    error = OSError("dataset unavailable")

    class UnavailableDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            raise error

    trainer = make_trainer(UnavailableDataset(), AsyncMock())
    async with assert_no_leaked_tasks():
        with pytest.raises(OSError) as caught:
            await asyncio.wait_for(trainer.fit_async(), timeout=5)
        assert caught.value is error
        trainer.backend.on_train_end.assert_awaited_once()


@pytest.mark.asyncio
async def test_training_failure_joins_pending_rollouts(make_trainer):
    started = set()
    stopped = set()
    peers_started = asyncio.Event()
    error = RuntimeError("training RPC failed")

    async def rollout(task, task_id, rollout_idx, result_idx):
        if task["slow"]:
            started.add(rollout_idx)
            if len(started) == 2:
                peers_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                stopped.add(rollout_idx)
        episode = Episode(
            id=f"{task_id}:{rollout_idx}",
            termination_reason=TerminationReason.ENV_DONE,
            trajectories=[Trajectory(name="agent", reward=rollout_idx, steps=[Step(prompt_ids=[1], response_ids=[2], logprobs=[-0.5])])],
        )
        return None, None, None, episode

    async def fail_training(state):
        await peers_started.wait()
        raise error

    trainer = make_trainer([{"slow": False}, {"slow": True}], rollout)
    trainer.backend.on_batch_start = fail_training
    async with assert_no_leaked_tasks():
        with pytest.raises(RuntimeError) as caught:
            await asyncio.wait_for(trainer.fit_async(), timeout=5)
        assert caught.value is error
        assert started == stopped == {0, 1}


@pytest.mark.asyncio
async def test_rollout_failure_interrupts_wait_for_blocked_peer(make_trainer):
    peer_started = asyncio.Event()
    peer_stopped = asyncio.Event()
    error = OSError("rollout RPC failed")

    async def rollout(task, task_id, rollout_idx, result_idx):
        if rollout_idx == 1:
            await peer_started.wait()
            raise error
        peer_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            peer_stopped.set()

    trainer = make_trainer([{}], rollout)
    async with assert_no_leaked_tasks():
        with pytest.raises(RuntimeError, match="Async rollout task failed") as caught:
            await asyncio.wait_for(trainer.fit_async(), timeout=5)
        assert caught.value.__cause__ is error
        assert peer_stopped.is_set()

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from omegaconf import OmegaConf

from rllm.data import Dataset, StatefulTaskDataLoader
from rllm.trainer.algorithms.config import (
    AlgorithmConfig,
    CompactFilteringConfig,
    RejectionSamplingConfig,
    TransformConfig,
)
from rllm.trainer.buffer import TaskBatch, TrajectoryGroupBuffer
from rllm.trainer.metrics_aggregator import MetricsAggregator
from rllm.trainer.sync_coordinator import SyncCoordinator, SyncCoordinatorConfig
from rllm.trainer.unified_trainer import TrainerState, UnifiedTrainer
from rllm.types import Episode, Step, Trajectory


def test_strict_sync_dataloader_is_taskwise_but_counts_optimizer_steps():
    trainer = object.__new__(UnifiedTrainer)
    trainer.train_dataset = Dataset([{"id": str(i)} for i in range(18)])
    trainer.val_dataset = None
    trainer.benchmark_dataset = None
    trainer.async_config = SimpleNamespace(enable=False)
    trainer.rs_config = SimpleNamespace(filter_uniform_groups=True)
    trainer.rllm_config = OmegaConf.create(
        {
            "data": {
                "train_batch_size": 8,
                "seed": 0,
                "val_batch_size": -1,
            },
            "rejection_sample": {"multiplier": 1},
            "trainer": {
                "total_epochs": 1,
                "total_batches": -1,
            },
        }
    )

    trainer._init_dataloaders()

    assert trainer._train_dataloader._batch_size == 1
    assert len(trainer._train_dataloader) == 18
    assert trainer._total_training_steps == 2


def test_strict_sync_collection_backfills_rejected_groups_without_overshoot():
    trainer = object.__new__(UnifiedTrainer)
    trainer.agent_workflow_engine = SimpleNamespace(set_training_step=Mock())

    async def generate_episodes(batch, **_kwargs):
        return [SimpleNamespace(task_id=item["id"]) for item in batch]

    trainer.backend = SimpleNamespace(generate_episodes=AsyncMock(side_effect=generate_episodes))

    class FakeBuffer:
        def __init__(self):
            self.items = []

        @property
        def ready_count(self):
            return len(self.items)

        async def add_episode(self, task_id, episode):
            if task_id != "reject":
                self.items.append(TaskBatch(groups=[], episodes=[episode]))

        async def get(self):
            return self.items.pop(0)

    buffer = FakeBuffer()
    coordinator = SimpleNamespace(
        on_group_dispatched=Mock(),
        on_group_consumed=Mock(),
    )
    state = TrainerState(global_step=3, epoch=0)
    task_iterator = iter(
        [
            [{"id": "reject"}],
            [{"id": "accept-1"}],
            [{"id": "accept-2"}],
            [{"id": "must-not-run"}],
        ]
    )

    batches, exhausted, candidate_count = asyncio.run(
        UnifiedTrainer._collect_strict_sync_task_batches(
            trainer,
            task_iterator=task_iterator,
            trainer_state=state,
            buffer=buffer,
            coordinator=coordinator,
            target_count=2,
        )
    )

    assert [batch.episodes[0].task_id for batch in batches] == [
        "accept-1",
        "accept-2",
    ]
    assert exhausted is False
    assert candidate_count == 3
    assert trainer.backend.generate_episodes.await_count == 2
    assert coordinator.on_group_dispatched.call_count == 3
    assert coordinator.on_group_consumed.call_count == 2


def test_group_buffer_filters_uniform_reward_and_accepts_mixed_reward():
    coordinator = SyncCoordinator(
        SyncCoordinatorConfig(
            mini_batch_size=1,
            group_size=2,
            staleness_threshold=0.0,
            trigger_parameter_sync_step=1,
            max_concurrent_rollouts=2,
        )
    )
    buffer = TrajectoryGroupBuffer(
        group_size=2,
        coordinator=coordinator,
        aggregator=MetricsAggregator(),
        algorithm_config=AlgorithmConfig(norm_adv_by_std_in_grpo=False),
        transform_config=TransformConfig(),
        cf_config=CompactFilteringConfig(),
        rs_config=RejectionSamplingConfig(filter_uniform_groups=True),
    )

    async def add_group(task_id, rewards):
        coordinator.on_group_dispatched()
        for rollout_idx, reward in enumerate(rewards):
            await buffer.add_episode(
                task_id,
                Episode(
                    id=f"{task_id}:{rollout_idx}",
                    is_correct=reward > 0,
                    trajectories=[
                        Trajectory(
                            reward=reward,
                            steps=[
                                Step(
                                    prompt_ids=[1],
                                    response_ids=[2],
                                    logprobs=[-0.1],
                                    reward=reward,
                                )
                            ],
                        )
                    ],
                ),
            )

    asyncio.run(add_group("uniform", [0.0, 0.0]))
    assert buffer.ready_count == 0

    asyncio.run(add_group("mixed", [0.0, 1.0]))
    assert buffer.ready_count == 1
    task_batch = asyncio.run(buffer.get())
    assert task_batch is not None
    assert [
        trajectory.steps[0].advantage
        for trajectory in task_batch.groups[0].trajectories
    ] == [-0.5, 0.5]


def test_strict_sync_accumulates_group_passes_then_optimizes_once():
    trainer = object.__new__(UnifiedTrainer)
    trainer.tokenizer = None

    pass_count = 0

    async def process_backend_batch(state):
        nonlocal pass_count
        pass_count += 1
        state.backend_batch = [object()]
        state.metrics["reward/default/mean"] = float(pass_count)

    async def update_policy(state):
        state.metrics["optim/grad_norm"] = 1.25

    trainer.backend = SimpleNamespace(
        on_batch_start=AsyncMock(),
        transform_to_backend_batch=Mock(return_value=[]),
        process_backend_batch=AsyncMock(side_effect=process_backend_batch),
        update_policy=AsyncMock(side_effect=update_policy),
    )
    state = TrainerState(global_step=1)
    batches = [
        TaskBatch(
            groups=[SimpleNamespace()],
            episodes=[SimpleNamespace(task_id=f"task-{i}")],
        )
        for i in range(8)
    ]

    trained = asyncio.run(
        UnifiedTrainer._train_strict_sync_task_batches(
            trainer,
            task_batches=batches,
            trainer_state=state,
            aggregator=MetricsAggregator(),
            candidate_count=11,
        )
    )

    assert trained is True
    assert trainer.backend.process_backend_batch.await_count == 8
    trainer.backend.update_policy.assert_awaited_once_with(state)
    assert state.policy_update_count == 1
    assert state.metrics["reward/default/mean"] == 4.5
    assert state.metrics["optim/grad_norm"] == 1.25
    assert state.metrics["sync/candidate_prompt_groups"] == 11
    assert state.metrics["sync/accepted_prompt_groups"] == 8
    assert state.metrics["sync/trainable_sequences"] == 8


def test_weight_sync_is_an_awaited_barrier_before_rollout_version_advances():
    events = []

    async def hotload(_state):
        events.append("hotload-complete")

    async def set_gateway_version(version):
        events.append(f"gateway-version-{version}")

    trainer = object.__new__(UnifiedTrainer)
    trainer.async_config = SimpleNamespace(partial_rollout=False)
    trainer.backend = SimpleNamespace(on_policy_updated=AsyncMock(side_effect=hotload))
    trainer._gateway = SimpleNamespace(aset_weight_version=AsyncMock(side_effect=set_gateway_version))

    rollout_engine = SimpleNamespace(weight_version=0)
    coordinator = SyncCoordinator(
        SyncCoordinatorConfig(
            mini_batch_size=8,
            group_size=16,
            staleness_threshold=0.0,
            trigger_parameter_sync_step=1,
            max_concurrent_rollouts=64,
        )
    )
    coordinator.on_training_step_complete()
    state = TrainerState(weight_version=0)

    asyncio.run(
        UnifiedTrainer._perform_weight_sync(
            trainer,
            state,
            coordinator,
            rollout_engine,
        )
    )

    assert events == ["hotload-complete", "gateway-version-1"]
    assert state.weight_version == 1
    assert rollout_engine.weight_version == 1
    assert coordinator.weight_version == 1


def test_strict_sync_loop_hotloads_before_next_rollout_collection():
    events = []
    trainer = object.__new__(UnifiedTrainer)
    trainer.rllm_config = OmegaConf.create(
        {
            "data": {"train_batch_size": 1},
            "rollout": {"n": 2},
            "trainer": {
                "total_epochs": 1,
                "total_batches": 2,
                "test_freq": -1,
            },
            "workflow": {"warm_queue_size": 0},
        }
    )
    trainer.async_config = SimpleNamespace(
        partial_rollout=False,
        episode_offload_dir=None,
        trajectory_group_offload_dir=None,
    )
    trainer.algorithm_config = AlgorithmConfig(norm_adv_by_std_in_grpo=False)
    trainer.transform_config = TransformConfig()
    trainer.cf_config = CompactFilteringConfig()
    trainer.rs_config = RejectionSamplingConfig(filter_uniform_groups=True)
    trainer.tokenizer = None
    trainer._gateway = None
    trainer._total_training_steps = 2
    trainer._train_dataloader = StatefulTaskDataLoader(
        Dataset([{"id": "task-1"}, {"id": "task-2"}]),
        batch_size=1,
        shuffle=False,
    )
    rollout_engine = SimpleNamespace(weight_version=0)
    trainer.agent_workflow_engine = SimpleNamespace(
        raise_on_error=False,
        n_parallel_tasks=2,
        rollout_engine=rollout_engine,
        hooks=None,
        set_training_step=Mock(),
    )

    async def generate_episodes(batch, **_kwargs):
        task_id = batch[0]["id"]
        events.append(f"generate-{task_id}")
        return [
            Episode(
                id=f"{task_id}:{rollout_idx}",
                is_correct=reward > 0,
                trajectories=[
                    Trajectory(
                        reward=reward,
                        steps=[
                            Step(
                                prompt_ids=[1],
                                response_ids=[2],
                                logprobs=[-0.1],
                                reward=reward,
                            )
                        ],
                    )
                ],
            )
            for rollout_idx, reward in enumerate([0.0, 1.0])
        ]

    async def process_backend_batch(state):
        events.append("fwd-bwd")
        state.backend_batch = [object()]

    async def update_policy(_state):
        events.append("optim")

    async def hotload(_state):
        events.append("hotload-complete")

    async def batch_end(_state):
        events.append("batch-end")

    trainer.backend = SimpleNamespace(
        on_epoch_start=AsyncMock(),
        on_epoch_end=AsyncMock(),
        generate_episodes=AsyncMock(side_effect=generate_episodes),
        on_batch_start=AsyncMock(),
        transform_to_backend_batch=Mock(return_value=[]),
        process_backend_batch=AsyncMock(side_effect=process_backend_batch),
        update_policy=AsyncMock(side_effect=update_policy),
        on_policy_updated=AsyncMock(side_effect=hotload),
        on_batch_end=AsyncMock(side_effect=batch_end),
    )
    trainer.logger = Mock()
    trainer._run_final_evaluations_async = AsyncMock(return_value={})
    state = TrainerState(
        global_step=1,
        train_dataloader=trainer._train_dataloader,
    )

    asyncio.run(UnifiedTrainer._fit_strict_sync_on_policy(trainer, state))

    assert events == [
        "generate-task-1",
        "fwd-bwd",
        "optim",
        "hotload-complete",
        "batch-end",
        "generate-task-2",
        "fwd-bwd",
        "optim",
        "hotload-complete",
        "batch-end",
    ]
    assert state.policy_update_count == 2
    assert state.weight_version == 2
    assert rollout_engine.weight_version == 2

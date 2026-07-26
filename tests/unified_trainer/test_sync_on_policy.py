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
from rllm.trainer.algorithms.transform import _default_traj_grouping_hook
from rllm.trainer.unified_trainer import TrainerState, UnifiedTrainer
from rllm.types import Episode, Step, TerminationReason, Trajectory


def test_sync_dataloader_keeps_the_optimizer_batch_intact():
    trainer = object.__new__(UnifiedTrainer)
    trainer.train_dataset = Dataset([{"id": str(i)} for i in range(18)])
    trainer.val_dataset = None
    trainer.benchmark_dataset = None
    trainer.async_config = SimpleNamespace(enable=False)
    # This option belongs to the async buffer. It must never change the
    # synchronous dataloader into a task-wise refill loop.
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

    assert trainer._train_dataloader._batch_size == 8
    assert len(trainer._train_dataloader) == 2
    assert trainer._total_training_steps == 2


def test_sync_loop_uses_one_full_batch_pass_and_hotloads_before_next_rollout():
    events: list[str] = []
    trainer = object.__new__(UnifiedTrainer)
    trainer.rllm_config = OmegaConf.create(
        {
            "data": {"train_batch_size": 2},
            "rollout": {"n": 2},
            "trainer": {
                "total_epochs": 1,
                "total_batches": 2,
                "test_freq": -1,
            },
            "workflow": {"warm_queue_size": 0},
        }
    )
    trainer.async_config = SimpleNamespace(enable=False)
    trainer.algorithm_config = AlgorithmConfig(norm_adv_by_std_in_grpo=True)
    trainer.transform_config = TransformConfig()
    trainer.cf_config = CompactFilteringConfig()
    trainer.rs_config = RejectionSamplingConfig(filter_uniform_groups=False)
    trainer.traj_grouping_hook = _default_traj_grouping_hook
    trainer.tokenizer = None
    trainer._total_training_steps = 2
    trainer._train_dataloader = StatefulTaskDataLoader(
        Dataset([{"id": f"task-{idx}"} for idx in range(1, 5)]),
        batch_size=2,
        shuffle=False,
    )
    trainer.agent_workflow_engine = SimpleNamespace(
        hooks=None,
        set_training_step=Mock(),
    )

    async def generate_episodes(batch, **_kwargs):
        task_ids = [item["id"] for item in batch]
        events.append(f"generate-{','.join(task_ids)}")
        return [
            Episode(
                id=f"{task_id}:{rollout_idx}",
                is_correct=bool(reward),
                termination_reason=TerminationReason.ENV_DONE,
                trajectories=[
                    Trajectory(
                        name="opencode",
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
            for task_id in task_ids
            for rollout_idx, reward in enumerate([0.0, 1.0])
        ]

    async def process_backend_batch(state):
        task_ids = [group.task_id for group in state.trajectory_groups]
        events.append(f"fwd-bwd-{','.join(task_ids)}")

    async def update_policy(_state):
        events.append("optim")

    async def hotload_at_batch_end(_state):
        events.append("hotload-complete")

    trainer.backend = SimpleNamespace(
        on_epoch_start=AsyncMock(),
        on_epoch_end=AsyncMock(),
        generate_episodes=AsyncMock(side_effect=generate_episodes),
        on_batch_start=AsyncMock(),
        transform_to_backend_batch=Mock(return_value=[object()]),
        process_backend_batch=AsyncMock(side_effect=process_backend_batch),
        compute_advantages=AsyncMock(),
        update_policy=AsyncMock(side_effect=update_policy),
        on_batch_end=AsyncMock(side_effect=hotload_at_batch_end),
    )
    trainer.logger = Mock()
    trainer._run_final_evaluations_async = AsyncMock(return_value={})
    state = TrainerState(
        global_step=1,
        train_dataloader=trainer._train_dataloader,
    )

    asyncio.run(UnifiedTrainer._fit_on_policy(trainer, state))

    assert events == [
        "generate-task-1,task-2",
        "fwd-bwd-task-1,task-2",
        "optim",
        "hotload-complete",
        "generate-task-3,task-4",
        "fwd-bwd-task-3,task-4",
        "optim",
        "hotload-complete",
    ]
    assert trainer.backend.process_backend_batch.await_count == 2
    assert trainer.backend.update_policy.await_count == 2
    assert state.policy_update_count == 2

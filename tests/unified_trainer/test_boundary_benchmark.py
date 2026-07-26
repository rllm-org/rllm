import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from rllm.data import Dataset
from rllm.trainer.algorithms.config import AlgorithmConfig, CompactFilteringConfig, TransformConfig
from rllm.trainer.algorithms.transform import _default_traj_grouping_hook
from rllm.trainer.unified_trainer import TrainerState, UnifiedTrainer
from rllm.types import Episode, Step, TerminationReason, Trajectory


def test_boundary_suites_commit_one_combined_metrics_row():
    trainer = object.__new__(UnifiedTrainer)
    trainer.logger = Mock()
    trainer._validate_async = AsyncMock(return_value={"val/reward/mean": 0.25})
    trainer._benchmark_async = AsyncMock(return_value={"benchmark/reward/mean": 0.5})
    state = TrainerState(global_step=0)

    metrics = asyncio.run(
        trainer._run_evaluation_suites_async(
            state,
            run_validation=True,
            run_benchmark=True,
        )
    )

    assert metrics == {
        "val/reward/mean": 0.25,
        "benchmark/reward/mean": 0.5,
    }
    trainer._validate_async.assert_awaited_once_with(state, log_metrics=False)
    trainer._benchmark_async.assert_awaited_once_with(state, log_metrics=False)
    trainer.logger.log.assert_called_once_with(data=metrics, step=0)


def test_final_evaluation_schedule_uses_periodic_suite_and_boundary_benchmark():
    trainer = object.__new__(UnifiedTrainer)
    trainer.rllm_config = OmegaConf.create(
        {
            "trainer": {
                "test_freq": 10,
                "benchmark_after_train": True,
            }
        }
    )
    trainer._run_evaluation_suites_async = AsyncMock(return_value={})
    state = TrainerState(global_step=150)

    asyncio.run(trainer._run_final_evaluations_async(state))

    trainer._run_evaluation_suites_async.assert_awaited_once_with(
        state,
        run_validation=True,
        run_benchmark=True,
    )


def test_final_evaluation_schedule_runs_validation_when_boundary_benchmark_is_disabled():
    trainer = object.__new__(UnifiedTrainer)
    trainer.rllm_config = OmegaConf.create(
        {
            "trainer": {
                "test_freq": 10,
                "benchmark_after_train": False,
            }
        }
    )
    trainer._run_evaluation_suites_async = AsyncMock(return_value={})
    state = TrainerState(global_step=149)

    asyncio.run(trainer._run_final_evaluations_async(state))

    trainer._run_evaluation_suites_async.assert_awaited_once_with(
        state,
        run_validation=True,
        run_benchmark=False,
    )


def test_final_evaluation_skips_policy_already_covered_by_periodic_validation():
    trainer = object.__new__(UnifiedTrainer)
    trainer.rllm_config = OmegaConf.create(
        {
            "trainer": {
                "test_freq": 10,
                "benchmark_after_train": False,
            }
        }
    )
    trainer._run_evaluation_suites_async = AsyncMock(return_value={})
    state = TrainerState(
        global_step=151,
        policy_update_count=150,
        last_validation_policy_update_count=150,
    )

    metrics = asyncio.run(trainer._run_final_evaluations_async(state))

    assert metrics == {}
    trainer._run_evaluation_suites_async.assert_not_awaited()


def test_validation_records_the_policy_update_it_evaluated():
    trainer = object.__new__(UnifiedTrainer)
    trainer._val_dataloader = Mock()
    trainer._evaluate_dataloader_async = AsyncMock(return_value={"val/reward/mean": 0.5})
    state = TrainerState(global_step=10, policy_update_count=10)

    metrics = asyncio.run(trainer._validate_async(state))

    assert metrics == {"val/reward/mean": 0.5}
    assert state.last_validation_policy_update_count == 10


def test_async_periodic_validation_can_defer_logging_for_training_row():
    trainer = object.__new__(UnifiedTrainer)
    trainer._validate_async = AsyncMock(return_value={"val/reward/mean": 0.25})
    coordinator = SimpleNamespace(
        pause_generation=Mock(),
        wait_for_drain=AsyncMock(),
        resume_generation=Mock(),
    )
    state = TrainerState(global_step=10)

    metrics = asyncio.run(
        trainer._validate_async_with_pause(
            state,
            coordinator,
            log_metrics=False,
        )
    )

    assert metrics == {"val/reward/mean": 0.25}
    coordinator.pause_generation.assert_called_once_with()
    coordinator.wait_for_drain.assert_awaited_once_with()
    trainer._validate_async.assert_awaited_once_with(state, log_metrics=False)
    coordinator.resume_generation.assert_called_once_with()


def test_init_dataloaders_builds_separate_validation_and_benchmark_loaders():
    trainer = object.__new__(UnifiedTrainer)
    trainer.train_dataset = None
    trainer.val_dataset = Dataset([{"task_id": "midtest-1"}, {"task_id": "midtest-2"}])
    trainer.benchmark_dataset = Dataset([{"task_id": f"tb21-{idx}"} for idx in range(89)])
    trainer.rllm_config = SimpleNamespace(data=SimpleNamespace(val_batch_size=-1))

    trainer._init_dataloaders()

    assert len(trainer._val_dataloader) == 1
    assert len(trainer._benchmark_dataloader) == 1
    assert len(next(iter(trainer._val_dataloader))) == 2
    assert len(next(iter(trainer._benchmark_dataloader))) == 89


def test_eval_reward_mean_keeps_every_attempt_in_the_denominator():
    episodes = [
        Episode(
            id="task-1:0",
            is_correct=True,
            termination_reason=TerminationReason.ENV_DONE,
            trajectories=[
                Trajectory(
                    name="opencode",
                    reward=1.0,
                    steps=[Step(reward=1.0)],
                )
            ],
            metadata={"data_source": "tb21"},
        ),
        Episode(
            id="task-2:0",
            is_correct=False,
            termination_reason=TerminationReason.ERROR,
            trajectories=[
                Trajectory(
                    name="opencode",
                    reward=0.0,
                    steps=[Step(reward=0.0)],
                )
            ],
            metadata={"data_source": "tb21"},
        ),
        Episode(
            id="task-3:0",
            is_correct=False,
            termination_reason=TerminationReason.MODEL_ERROR,
            trajectories=[],
            metadata={"data_source": "tb21"},
        ),
    ]

    trainer = object.__new__(UnifiedTrainer)
    trainer.rllm_config = SimpleNamespace(rollout=SimpleNamespace(n_val=1))
    trainer.algorithm_config = AlgorithmConfig()
    trainer.transform_config = TransformConfig()
    trainer.cf_config = CompactFilteringConfig(
        enable=True,
        mask_error=True,
        mask_model_error=True,
    )
    trainer.traj_grouping_hook = _default_traj_grouping_hook
    trainer.agent_workflow_engine = SimpleNamespace(set_training_step=Mock())
    trainer.backend = SimpleNamespace(
        on_validation_start=AsyncMock(return_value=True),
        generate_episodes=AsyncMock(return_value=episodes),
        on_validation_end=AsyncMock(),
    )
    trainer.logger = Mock()

    metrics = asyncio.run(
        trainer._evaluate_dataloader_async(
            TrainerState(global_step=0),
            dataloader=[[{"id": "unused"}]],
            metric_prefix="val",
            title="Validation",
            log_metrics=False,
        )
    )

    assert metrics["val/reward/mean"] == pytest.approx(1.0 / 3.0)
    assert metrics["val/reward/num_episodes"] == 3
    assert metrics["val/reward/num_correct"] == 1
    assert metrics["val/tb21/pass@1"] == pytest.approx(1.0 / 3.0)
    # The role-specific metric is intentionally a scorable-trajectory
    # diagnostic. Compact filtering removes the two invalid attempts from it.
    assert metrics["val/reward/opencode/mean"] == 1.0

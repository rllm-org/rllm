import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from omegaconf import OmegaConf

from rllm.data import Dataset
from rllm.trainer.unified_trainer import TrainerState, UnifiedTrainer


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

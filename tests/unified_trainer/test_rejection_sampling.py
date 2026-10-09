"""CPU regressions for rejection-sampling state across on-policy batches."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from rllm.trainer.algorithms.config import AlgorithmConfig, CompactFilteringConfig, RejectionSamplingConfig, TransformConfig
from rllm.trainer.algorithms.rejection_sampling import RejectionSamplingState, apply_rejection_sampling_and_filtering
from rllm.trainer.algorithms.transform import _default_traj_grouping_hook, transform_episodes_to_trajectory_groups
from rllm.trainer.unified_trainer import TrainerState, UnifiedTrainer
from rllm.types import Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationReason


def _episodes(task_id, correctness):
    return [
        Episode(
            id=f"{task_id}:{index}",
            is_correct=correct,
            termination_reason=TerminationReason.ENV_DONE,
            trajectories=[Trajectory(name="solver", reward=float(correct), steps=[Step()])],
        )
        for index, correct in enumerate(correctness)
    ]


class _BatchLoader(list):
    epoch = 0


@pytest.mark.parametrize(
    "mode,threshold,batches,expected_updates",
    [
        pytest.param("episode", 2, [(False, True)] * 4, [[0, 1], [2, 3]], id="two-accumulations"),
        pytest.param("episode", 2, [(False, True), (), (True,), (False, True)], [[0, 3]], id="empty-and-filtered-batches"),
        pytest.param("episode", 2, [(False, True), (True, True), (False, False), (False, True)], [[0, 1, 2, 3]], id="uniform-groups-do-not-meet-threshold"),
        pytest.param("episode", 1, [(False, True)] * 3, [[0], [1], [2]], id="immediate-release"),
        pytest.param("none", 2, [(True, True), (False, False), (False, True)], [[0], [1], [2]], id="no-rejection"),
    ],
)
def test_on_policy_updates_only_after_sampling_releases_groups(mode, threshold, batches, expected_updates):
    """Exercise the actual loop, transform and filter with a CPU-only backend."""
    updates = []

    def record_update(state):
        updates.append(
            {
                "groups": [group.group_id for group in state.trajectory_groups],
                "episodes": [episode.id for episode in state.episodes if episode.trajectories],
                "metrics": state.metrics.copy(),
            }
        )

    trainer = UnifiedTrainer.__new__(UnifiedTrainer)
    trainer._train_dataloader = _BatchLoader(_episodes(f"task-{index}", correctness) for index, correctness in enumerate(batches))
    trainer._total_training_steps = len(batches)
    trainer.rllm_config = OmegaConf.create({"trainer": {"total_epochs": 1, "test_freq": 0}, "workflow": {"warm_queue_size": 0}})
    trainer.agent_workflow_engine = SimpleNamespace(hooks=None, set_training_step=Mock())
    trainer.backend = SimpleNamespace(
        on_epoch_start=AsyncMock(),
        on_epoch_end=AsyncMock(),
        on_batch_start=AsyncMock(),
        on_batch_end=AsyncMock(),
        generate_episodes=AsyncMock(side_effect=lambda batch, **kwargs: batch),
        transform_to_backend_batch=Mock(side_effect=lambda state: state.trajectory_groups),
        process_backend_batch=AsyncMock(),
        compute_advantages=AsyncMock(),
        update_policy=AsyncMock(side_effect=record_update),
    )
    trainer.logger = SimpleNamespace(log=Mock())
    trainer.transform_config = TransformConfig()
    trainer.cf_config = CompactFilteringConfig()
    trainer.traj_grouping_hook = _default_traj_grouping_hook
    trainer.rs_config = RejectionSamplingConfig(mode=mode, min_partial_solve_tasks=threshold)
    trainer.algorithm_config = AlgorithmConfig()
    trainer.tokenizer = None

    asyncio.run(trainer._fit_on_policy(TrainerState(global_step=1)))

    assert [update["groups"] for update in updates] == [[f"task-{index}:solver" for index in indexes] for indexes in expected_updates]
    for update, indexes in zip(updates, expected_updates, strict=True):
        assert update["episodes"] == [f"task-{index}:{rollout}" for index in indexes for rollout in range(2)]
        assert update["metrics"]["batch/num_tasks"] == len(indexes)
        partial = sum(any(batches[index]) and not all(batches[index]) for index in indexes)
        assert update["metrics"]["batch/solve_partial"] == pytest.approx(partial / len(indexes))
    assert trainer.backend.update_policy.await_count == len(expected_updates)


def test_reset_batch_preserves_pending_samples_and_clears_per_batch_fields():
    episodes = _episodes("pending", (False, True))
    groups, _ = transform_episodes_to_trajectory_groups(episodes, TransformConfig())
    state = TrainerState(timing_dict={"generation": 1}, metrics={"previous": 1}, extra_info={"previous": 1}, backend_batch=object())
    config = RejectionSamplingConfig(mode="episode", min_partial_solve_tasks=2)
    state.trajectory_groups, state.episodes, _ = apply_rejection_sampling_and_filtering(episodes, groups, config, state.rs_state)

    state.reset_batch()

    assert state.rs_state.accumulated_groups == groups
    assert state.rs_state.accumulated_episodes == episodes
    assert state.rs_state.metrics.solve_partial == 1
    assert state.episodes is None
    assert state.trajectory_groups is None
    assert state.backend_batch is None
    assert state.timing_dict == state.metrics == state.extra_info == {}


@pytest.mark.parametrize("mode", ["episode", "none"])
def test_release_clears_state_and_preserves_returned_data_and_metrics(mode):
    config = RejectionSamplingConfig(mode=mode, min_partial_solve_tasks=1)
    state = RejectionSamplingState()
    first_episodes = _episodes("first", (False, True))
    first_groups, _ = transform_episodes_to_trajectory_groups(first_episodes, TransformConfig())

    released_groups, released_episodes, metrics = apply_rejection_sampling_and_filtering(first_episodes, first_groups, config, state)

    assert state.accumulated_groups == state.accumulated_episodes == []
    assert state.metrics.solve_partial == state.metrics.groups_before_filter == 0
    assert released_groups == first_groups
    assert released_episodes == first_episodes
    assert metrics["batch/num_tasks"] == 1
    assert metrics["batch/solve_partial"] == 1

    second_episodes = _episodes("second", (False, True))
    second_groups, _ = transform_episodes_to_trajectory_groups(second_episodes, TransformConfig())
    next_groups, next_episodes, _ = apply_rejection_sampling_and_filtering(second_episodes, second_groups, config, state)

    assert next_groups == second_groups
    assert next_episodes == second_episodes
    assert released_groups == first_groups
    assert released_episodes == first_episodes
    assert metrics["batch/num_tasks"] == 1

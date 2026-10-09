from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from rllm.trainer.fireworks.fireworks_backend import FireworksBackend
from rllm.trainer.sync_coordinator import SyncCoordinator, SyncCoordinatorConfig
from rllm.trainer.tinker.tinker_backend import TinkerBackend
from rllm.trainer.unified_trainer import TrainerState


def make_backend(backend_type, async_enabled=True, save_freq=4):
    backend = object.__new__(backend_type)
    backend.full_config = OmegaConf.create({"rllm": {"trainer": {"save_freq": save_freq}, "async_training": {"enable": async_enabled, "trigger_parameter_sync_step": 4}}})
    backend.learning_rate = 1e-6
    backend._policy_updated_this_step = False
    backend.rollout_engine = SimpleNamespace(set_sampling_client=Mock())
    backend.policy_trainer = SimpleNamespace(
        save_checkpoint_and_get_sampling_client=AsyncMock(side_effect=lambda step, **kwargs: step),
        sync_weights=AsyncMock(side_effect=lambda step, **kwargs: f"snapshot-{step}"),
        save_dcp_checkpoint=AsyncMock(),
        promote_checkpoint=AsyncMock(),
    )
    return backend


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_type", [TinkerBackend, FireworksBackend])
@pytest.mark.parametrize("async_enabled", [True, False], ids=["async", "sync-control"])
async def test_sync_cadence_preserves_checkpoints_after_resume(backend_type, async_enabled):
    backend = make_backend(backend_type, async_enabled)
    coordinator = SyncCoordinator(SyncCoordinatorConfig(mini_batch_size=1, group_size=2, staleness_threshold=0, trigger_parameter_sync_step=4))
    state = TrainerState(total_steps=9, train_dataloader=SimpleNamespace(state_dict=lambda: {"cursor": 8}))

    # Resuming after step 1 puts saves at 4/8 between coordinated syncs at 5/9.
    for step in range(2, 10):
        state.global_step = step
        if async_enabled:
            coordinator.on_training_step_complete()
            if coordinator.should_sync():
                await backend.on_policy_updated(state)
                coordinator.on_sync_complete()
        await backend.on_batch_end(state)

    policy = backend.policy_trainer
    expected_syncs = [5, 9] if async_enabled else list(range(2, 10))
    if backend_type is TinkerBackend:
        assert [call.args[0] for call in backend.rollout_engine.set_sampling_client.call_args_list] == expected_syncs
        saves = [call for call in policy.save_checkpoint_and_get_sampling_client.await_args_list if call.kwargs["do_save"]]
        assert [call.args[0] for call in saves] == [4, 8]
        assert [call.kwargs["dataloader_state"] for call in saves] == [{"cursor": 8}, {"cursor": 8}]
    else:
        assert [call.args[0] for call in policy.sync_weights.await_args_list] == expected_syncs
        assert [call.args[0] for call in policy.save_dcp_checkpoint.await_args_list] == [4, 8]
        assert [call.args[0] for call in policy.promote_checkpoint.await_args_list] == ([] if async_enabled else ["snapshot-4", "snapshot-8"])


@pytest.mark.asyncio
async def test_tinker_saves_final_checkpoint_with_periodic_saves_disabled():
    backend = make_backend(TinkerBackend, save_freq=-1)
    state = TrainerState(global_step=5, train_dataloader=SimpleNamespace(state_dict=lambda: {"cursor": 8}))
    await backend.on_train_end(state)
    backend.policy_trainer.save_checkpoint_and_get_sampling_client.assert_awaited_once_with(5, do_save=True, dataloader_state={"cursor": 8})
    backend.rollout_engine.set_sampling_client.assert_not_called()

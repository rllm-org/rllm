import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from omegaconf import OmegaConf

import rllm.trainer.unified_trainer as trainer_module
from rllm.trainer.unified_trainer import TrainerState, UnifiedTrainer


class FakeInferenceClient:
    pass


class FakeBackend:
    def __init__(self) -> None:
        self.gateway_inference_client = Mock(return_value=(FakeInferenceClient, {"weight_version": 0}))
        self.gateway_inference_client_update = Mock(side_effect=lambda version: {"weight_version": version})
        self.on_train_start = AsyncMock()
        self.on_train_end = AsyncMock()
        self.on_policy_updated = AsyncMock()


class FakeGateway:
    def __init__(self) -> None:
        self.start = Mock()
        self.astop = AsyncMock()
        self.update_inference_client = AsyncMock()


def test_fit_starts_v2_with_backend_inference_client_and_always_stops(monkeypatch) -> None:
    monkeypatch.setattr(trainer_module, "print_config_table", lambda *args, **kwargs: None)
    trainer = UnifiedTrainer.__new__(UnifiedTrainer)
    trainer.config = OmegaConf.create({})
    trainer.rllm_config = OmegaConf.create({"trainer": {"val_before_train": False, "val_only": False}})
    trainer.backend = FakeBackend()
    trainer._gateway = FakeGateway()
    trainer._gateway_version = "v2"
    trainer._remote_runtime = None
    trainer.agent_workflow_engine = SimpleNamespace()
    trainer._train_dataloader = None
    trainer._fit_async = AsyncMock()

    asyncio.run(trainer.fit_async())

    trainer.backend.gateway_inference_client.assert_called_once_with(0)
    trainer._gateway.start.assert_called_once_with(
        FakeInferenceClient,
        {"weight_version": 0},
    )
    trainer._fit_async.assert_awaited_once()
    trainer._gateway.astop.assert_awaited_once_with()
    trainer.backend.on_train_end.assert_awaited_once()


def test_async_weight_sync_updates_v2_inference_clients() -> None:
    trainer = UnifiedTrainer.__new__(UnifiedTrainer)
    trainer.backend = FakeBackend()
    trainer._gateway = FakeGateway()
    trainer._gateway_version = "v2"
    trainer.async_config = SimpleNamespace(partial_rollout=True)
    state = TrainerState(weight_version=2)
    coordinator = SimpleNamespace(weight_version=6, on_sync_complete=Mock())

    asyncio.run(trainer._perform_weight_sync(state, coordinator, rollout_engine=None))

    assert state.weight_version == 7
    trainer.backend.on_policy_updated.assert_awaited_once_with(state)
    trainer.backend.gateway_inference_client_update.assert_called_once_with(7)
    trainer._gateway.update_inference_client.assert_awaited_once_with({"weight_version": 7})
    coordinator.on_sync_complete.assert_called_once_with()

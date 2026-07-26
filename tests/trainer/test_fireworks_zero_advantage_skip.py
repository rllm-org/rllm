import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from rllm.trainer.fireworks.fireworks_backend import FireworksBackend
from rllm.trainer.unified_trainer import TrainerState


def test_skipped_policy_update_does_not_hotload_fireworks_deployment():
    backend = object.__new__(FireworksBackend)
    backend.policy_trainer = SimpleNamespace(
        _compute_rollout_entropy_metrics=Mock(return_value={}),
    )
    backend._policy_updated_this_step = False
    backend.learning_rate = 1e-6
    backend._save_and_sync = AsyncMock()
    state = TrainerState(
        global_step=7,
        total_steps=150,
        policy_updated_this_batch=False,
    )

    asyncio.run(FireworksBackend.on_batch_end(backend, state))

    backend._save_and_sync.assert_not_awaited()
    assert state.metrics["progress/batch"] == 7
    assert state.metrics["progress/lr"] == 1e-6

"""Wiring tests for the training warm queue (rllm/trainer/unified_trainer.py).

Cover the pieces that don't create real sandboxes: the validation detach
context manager and the guard that decides whether a queue is built at all.
"""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rllm.trainer.unified_trainer import UnifiedTrainer, _detached_warm_queue


def test_detached_warm_queue_detaches_and_restores():
    # Normal path: detached inside the block, restored after.
    hooks = SimpleNamespace(warm_queue="Q")
    with _detached_warm_queue(hooks):
        assert hooks.warm_queue is None
    assert hooks.warm_queue == "Q"

    # Exception path: restored via finally.
    with pytest.raises(RuntimeError):
        with _detached_warm_queue(hooks):
            raise RuntimeError("boom")
    assert hooks.warm_queue == "Q"

    # None hooks: no-op, must not raise.
    with _detached_warm_queue(None):
        pass


def _fake_trainer(warm_queue_size, sandbox_backend, hooks_present=True):
    rllm_config = OmegaConf.create(
        {
            "workflow": {"warm_queue_size": warm_queue_size},
            "rollout": {"n": 8},
        }
    )
    hooks = SimpleNamespace(sandbox_backend=sandbox_backend, registry=None, warm_queue=None) if hooks_present else None
    engine = SimpleNamespace(hooks=hooks, agent_flow=SimpleNamespace(max_concurrent=4))
    return SimpleNamespace(rllm_config=rllm_config, agent_workflow_engine=engine, _total_training_steps=10), hooks


@pytest.mark.parametrize(
    ("warm_queue_size", "sandbox_backend", "hooks_present"),
    [
        pytest.param(0, "daytona", True, id="disabled"),
        pytest.param(-1, None, True, id="without_backend"),
        pytest.param(-1, "daytona", False, id="without_hooks"),
    ],
)
def test_start_warm_queue_guard_returns_none(warm_queue_size, sandbox_backend, hooks_present):
    """Each clause of the guard if independently disables the queue."""
    trainer, hooks = _fake_trainer(warm_queue_size=warm_queue_size, sandbox_backend=sandbox_backend, hooks_present=hooks_present)
    result = UnifiedTrainer._start_train_warm_queue(trainer, None, SimpleNamespace(global_step=1), 2, False, hooks)
    assert result is None
    if hooks is not None:
        assert hooks.warm_queue is None

"""Wiring tests for the training warm queue (rllm/trainer/unified_trainer.py).

Cover the pieces that don't create real sandboxes: the validation detach
context manager and the guard that decides whether a queue is built at all.
"""

from types import SimpleNamespace

from omegaconf import OmegaConf

from rllm.trainer.unified_trainer import UnifiedTrainer, _detached_warm_queue


def test_detached_warm_queue_restores():
    hooks = SimpleNamespace(warm_queue="Q")
    with _detached_warm_queue(hooks):
        assert hooks.warm_queue is None
    assert hooks.warm_queue == "Q"


def test_detached_warm_queue_restores_on_exception():
    hooks = SimpleNamespace(warm_queue="Q")
    try:
        with _detached_warm_queue(hooks):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert hooks.warm_queue == "Q"


def test_detached_warm_queue_handles_none_hooks():
    with _detached_warm_queue(None):
        pass  # must not raise


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


def test_start_warm_queue_returns_none_when_disabled():
    trainer, hooks = _fake_trainer(warm_queue_size=0, sandbox_backend="daytona")
    result = UnifiedTrainer._start_train_warm_queue(trainer, None, SimpleNamespace(global_step=1), 2, False, hooks)
    assert result is None
    assert hooks.warm_queue is None


def test_start_warm_queue_returns_none_without_backend():
    trainer, hooks = _fake_trainer(warm_queue_size=-1, sandbox_backend=None)
    result = UnifiedTrainer._start_train_warm_queue(trainer, None, SimpleNamespace(global_step=1), 2, False, hooks)
    assert result is None


def test_start_warm_queue_returns_none_without_hooks():
    trainer, _ = _fake_trainer(warm_queue_size=-1, sandbox_backend="daytona", hooks_present=False)
    result = UnifiedTrainer._start_train_warm_queue(trainer, None, SimpleNamespace(global_step=1), 2, False, None)
    assert result is None

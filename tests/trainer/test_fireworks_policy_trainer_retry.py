"""Transient-fault tolerance for Fireworks training-client RPCs.

Covers ``FireworksPolicyTrainer._run_training_op`` / ``_is_transient`` /
``_reconnect_training_client`` in isolation (the trainer is built via
``__new__`` so the heavy SDK init is skipped). Run via ``asyncio.run`` — no
pytest-asyncio needed.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rllm.trainer.fireworks.fireworks_policy_trainer import FireworksPolicyTrainer


class _FakeClient:
    def __init__(self):
        self.endpoints = []

    def _use_endpoint(self, ep):
        self.endpoints.append(ep)


class _FakeMgr:
    def wait_for_existing(self, job_id):
        return f"ep-{job_id}"


def _trainer(*, max_retries=2, backoff=0.0, rlor_mgr=None, job_id="job-1", client=None):
    """A trainer with only the attributes the fault-tolerance helpers touch."""
    t = FireworksPolicyTrainer.__new__(FireworksPolicyTrainer)
    t._step_max_retries = max_retries
    t._step_retry_backoff_s = backoff
    t._rlor_mgr = rlor_mgr
    t._policy_job_id = job_id
    t.training_client = client
    return t


def test_is_transient_classification():
    t = _trainer()
    assert t._is_transient(TimeoutError("timed out"))
    assert t._is_transient(ConnectionError("reset"))
    assert t._is_transient(RuntimeError("StatusCode.UNAVAILABLE"))
    assert t._is_transient(RuntimeError("deadline exceeded"))
    assert t._is_transient(RuntimeError("upstream returned 503"))
    # Data / programming errors are NOT transient — must propagate.
    assert not t._is_transient(ValueError("malformed datum"))
    assert not t._is_transient(KeyError("missing field"))


def test_retry_succeeds_after_transient_blips():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("blip")
        return "ok"

    t = _trainer(max_retries=3)
    assert asyncio.run(t._run_training_op(fn, op_name="x")) == "ok"
    assert len(calls) == 3  # 2 failures + 1 success


def test_retry_exhausted_reraises():
    def fn():
        raise TimeoutError("persistent")

    t = _trainer(max_retries=2)
    with pytest.raises(TimeoutError):
        asyncio.run(t._run_training_op(fn, op_name="x"))


def test_non_transient_not_retried():
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("bad data")

    t = _trainer(max_retries=5)
    with pytest.raises(ValueError):
        asyncio.run(t._run_training_op(fn, op_name="x"))
    assert len(calls) == 1  # failed once, no retry


def test_zero_retries_disables():
    calls = []

    def fn():
        calls.append(1)
        raise TimeoutError("blip")

    t = _trainer(max_retries=0)
    with pytest.raises(TimeoutError):
        asyncio.run(t._run_training_op(fn, op_name="x"))
    assert len(calls) == 1


def test_reconnect_invoked_between_retries_when_enabled():
    client = _FakeClient()
    t = _trainer(max_retries=2, rlor_mgr=_FakeMgr(), job_id="j", client=client)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise TimeoutError("blip")
        return "ok"

    assert asyncio.run(t._run_training_op(fn, op_name="x", reconnect=True)) == "ok"
    assert client.endpoints == ["ep-j"]  # same-job channel refresh, once


def test_reconnect_skipped_when_disabled():
    client = _FakeClient()
    t = _trainer(max_retries=2, rlor_mgr=_FakeMgr(), client=client)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise TimeoutError("blip")
        return "ok"

    asyncio.run(t._run_training_op(fn, op_name="x", reconnect=False))
    assert client.endpoints == []  # never reconnected


def test_reconnect_without_mgr_is_safe_noop():
    # _reconnect returns False (no rlor_mgr) but the retry loop still proceeds.
    t = _trainer(max_retries=1, rlor_mgr=None)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise TimeoutError("blip")
        return "ok"

    assert asyncio.run(t._run_training_op(fn, op_name="x", reconnect=True)) == "ok"
    assert t._reconnect_training_client() is False


def test_initialize_from_scratch_awaits_initial_weight_hotload():
    t = _trainer()
    t.config = SimpleNamespace(model=SimpleNamespace(name="base-model"))
    t._try_resume = AsyncMock(return_value=0)
    t._initial_weight_sync = AsyncMock()

    start_step = asyncio.run(t.initialize_async(resume_from_checkpoint=True))

    assert start_step == 0
    t._try_resume.assert_awaited_once_with()
    t._initial_weight_sync.assert_awaited_once_with()


def test_initialize_after_resume_does_not_repeat_initial_weight_hotload():
    t = _trainer()
    t.config = SimpleNamespace(model=SimpleNamespace(name="base-model"))
    t._try_resume = AsyncMock(return_value=7)
    t._initial_weight_sync = AsyncMock()

    start_step = asyncio.run(t.initialize_async(resume_from_checkpoint=True))

    assert start_step == 7
    t._initial_weight_sync.assert_not_awaited()


def test_initial_weight_sync_uses_base_snapshot_and_is_awaited():
    t = _trainer()
    t._sync_weights = AsyncMock(return_value="snapshot")

    asyncio.run(t._initial_weight_sync())

    t._sync_weights.assert_awaited_once_with(
        "step-0-base",
        checkpoint_type="base",
    )


def test_optim_step_disables_server_normalization_and_emits_grad_norm_metrics():
    from fireworks.training.sdk.client import GradAccNormalization

    captured = {}

    def optim_step(adam_params, *, grad_accumulation_normalization):
        captured["adam_params"] = adam_params
        captured["normalization"] = grad_accumulation_normalization
        return SimpleNamespace(metrics={})

    t = _trainer(max_retries=0, client=SimpleNamespace(optim_step=optim_step))
    t.algorithm_config = SimpleNamespace(lr_schedule="constant", warmup_steps_ratio=0.0)
    t._custom_loss_sum = 0.0
    t._custom_loss_tokens = 0.0

    scheduled_lr, metrics = asyncio.run(
        t.optim_step(
            step=1,
            total_steps=10,
            learning_rate=1e-6,
        )
    )

    assert scheduled_lr == 1e-6
    assert metrics == {}
    assert captured["adam_params"].emit_grad_norm_metrics is True
    assert captured["normalization"] == GradAccNormalization.NONE

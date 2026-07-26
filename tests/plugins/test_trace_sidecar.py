"""U.5-U.6 — unit tests for rllm_trace_sidecar."""

from __future__ import annotations

import os


def test_atomic_write(tmp_path, monkeypatch) -> None:
    """U.5: _atomic_write uses .tmp + rename."""
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))

    # Re-import to pick up new RLLM_HOME
    import importlib
    import rllm_trace_sidecar

    importlib.reload(rllm_trace_sidecar)

    target = str(tmp_path / "out.txt")
    rllm_trace_sidecar._atomic_write(target, "42")

    with open(target) as f:
        assert f.read() == "42"
    assert not os.path.exists(target + ".tmp")


def test_monkey_patch_installed(monkeypatch, tmp_path) -> None:
    """U.6: after import, VerlBackend.on_batch_start is wrapped (identity changes).

    We patch out `rllm.trainer.verl.verl_backend` before importing the plugin so
    the test doesn't require the real verl stack to be importable.
    """
    import sys
    import types

    # Fake VerlBackend module — plugin will patch this fake.
    class _FakeState:
        global_step = 0

    async def _fake_on_train_start(self, s):
        return "orig_train_start"

    async def _fake_on_batch_start(self, s):
        return "orig_batch_start"

    async def _fake_on_policy_updated(self, s):
        return "orig_policy_updated"

    fake_backend = type(
        "VerlBackend",
        (),
        {
            "on_train_start": _fake_on_train_start,
            "on_batch_start": _fake_on_batch_start,
            "on_policy_updated": _fake_on_policy_updated,
        },
    )

    # Build the fake `rllm.trainer.verl.verl_backend` module chain
    fake_module = types.ModuleType("rllm.trainer.verl.verl_backend")
    fake_module.VerlBackend = fake_backend  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rllm", types.ModuleType("rllm"))
    monkeypatch.setitem(sys.modules, "rllm.trainer", types.ModuleType("rllm.trainer"))
    monkeypatch.setitem(sys.modules, "rllm.trainer.verl", types.ModuleType("rllm.trainer.verl"))
    monkeypatch.setitem(sys.modules, "rllm.trainer.verl.verl_backend", fake_module)
    monkeypatch.setenv("RLLM_HOME", str(tmp_path))

    # Now reload the plugin
    import importlib
    import rllm_trace_sidecar

    importlib.reload(rllm_trace_sidecar)

    # Verify hooks got replaced
    assert fake_backend.on_train_start is not _fake_on_train_start
    assert fake_backend.on_batch_start is not _fake_on_batch_start
    assert fake_backend.on_policy_updated is not _fake_on_policy_updated

    # Verify the patched on_batch_start writes current_step.txt
    import asyncio

    state = _FakeState()
    state.global_step = 7
    asyncio.run(fake_backend.on_batch_start(fake_backend(), state))

    step_path = tmp_path / "observability" / "current_step.txt"
    assert step_path.exists()
    assert step_path.read_text() == "7"

"""Tests for the Daytona sandbox backend wrapper.

Covers behavior that doesn't require the real ``daytona`` SDK (or a
DAYTONA_API_KEY): the friendly error when the SDK isn't installed, plus the
pure command-building / error-classification helpers and the exec contract
(``su`` user-switch, persistent-env export, timeout → SandboxCommandTimeout)
and the create retry loop, exercised against a fake sandbox/client.
"""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

import rllm.sandbox.backends.daytona as daytona_mod
from rllm.sandbox.backends.daytona import (
    DaytonaSandbox,
    _build_exec_command,
    _CreateRateLimiter,
    _is_transient_daytona_error,
    _looks_like_timeout,
)
from rllm.sandbox.protocol import SandboxCommandTimeout


def test_missing_daytona_sdk_raises_friendly_install_hint(monkeypatch):
    """When the ``daytona`` package isn't installed, instantiating
    DaytonaSandbox should raise an ImportError naming the install command,
    not a bare ``ModuleNotFoundError("No module named 'daytona'")``.
    """
    # Drop any cached daytona module and intercept the lazy import.
    monkeypatch.delitem(sys.modules, "daytona", raising=False)
    real_import = builtins.__import__

    def _block_daytona(name, *args, **kwargs):
        if name == "daytona" or name.startswith("daytona."):
            raise ModuleNotFoundError("No module named 'daytona'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_daytona)

    with pytest.raises(ImportError, match="pip install daytona"):
        DaytonaSandbox(name="test")


# ---------------------------------------------------------------------------
# _build_exec_command — persistent-env export + su user-switch (P0-1, P0-3)
# ---------------------------------------------------------------------------


def test_build_exec_command_exports_persistent_env():
    cmd = _build_exec_command("echo hi", {"FOO": "bar baz"}, None)
    assert cmd == "export FOO='bar baz'; echo hi"


def test_build_exec_command_su_for_str_user():
    cmd = _build_exec_command("echo hi", None, "agent")
    assert cmd == "su agent -s /bin/bash -c 'echo hi'"


def test_build_exec_command_resolves_int_uid_and_keeps_env_inside_switch():
    cmd = _build_exec_command("echo hi", {"K": "v"}, 1000)
    assert cmd.startswith("su $(getent passwd 1000 | cut -d: -f1) -s /bin/bash -c ")
    assert "export K=v; echo hi" in cmd


def test_build_exec_command_noop_without_env_or_user():
    assert _build_exec_command("echo hi", None, None) == "echo hi"


# ---------------------------------------------------------------------------
# error classification helpers (P0-2, P1-4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg",
    ["429 Too Many Requests", "Rate limit exceeded", "no capacity, try again", "503 Service Unavailable"],
)
def test_transient_errors_are_retryable(msg):
    assert _is_transient_daytona_error(Exception(msg))


@pytest.mark.parametrize("msg", ["Snapshot foo not found", "invalid resources", "bad request"])
def test_non_transient_errors_are_not_retryable(msg):
    assert not _is_transient_daytona_error(Exception(msg))


def test_looks_like_timeout():
    assert _looks_like_timeout(Exception("operation timed out"))
    assert _looks_like_timeout(Exception("Timeout waiting for response"))
    assert not _looks_like_timeout(Exception("connection refused"))


# ---------------------------------------------------------------------------
# rate limiter (P1-4)
# ---------------------------------------------------------------------------


def test_rate_limiter_disabled_is_noop():
    _CreateRateLimiter(rate=0, burst=0).acquire()  # must not block or raise


def test_rate_limiter_absorbs_burst_then_throttles(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(daytona_mod.time, "sleep", lambda s: slept.append(s))
    lim = _CreateRateLimiter(rate=4.0, burst=3.0)
    for _ in range(3):
        lim.acquire()  # burst capacity — no throttle
    assert slept == []
    lim.acquire()  # 4th outruns the bucket → must wait
    assert slept and slept[0] > 0


# ---------------------------------------------------------------------------
# exec contract against a fake sandbox (P0-1/2/3)
# ---------------------------------------------------------------------------


def _make_sandbox(exec_impl):
    """A DaytonaSandbox bypassing __init__, wired to a fake process.exec."""
    sb = DaytonaSandbox.__new__(DaytonaSandbox)
    sb.name = "test"
    sb._closed = False
    sb._persistent_env = {}
    sb._sandbox = SimpleNamespace(process=SimpleNamespace(exec=exec_impl))
    return sb


def test_exec_wraps_timeout_and_sets_sdk_backstop():
    seen = {}

    def fake_exec(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return SimpleNamespace(result="ok", exit_code=0)

    sb = _make_sandbox(fake_exec)
    assert sb.exec("echo hi", timeout=30) == "ok"
    assert seen["command"].startswith("timeout -k 10 30 bash -c ")
    assert seen["kwargs"]["timeout"] == 90  # 30 + 60s SDK backstop


@pytest.mark.parametrize("code", [124, 137])
def test_exec_timeout_exit_codes_raise_command_timeout(code):
    sb = _make_sandbox(lambda command, **kw: SimpleNamespace(result="partial", exit_code=code))
    with pytest.raises(SandboxCommandTimeout):
        sb.exec("sleep 999", timeout=5)


def test_exec_124_without_timeout_is_plain_failure():
    # No timeout set → 124 is just a non-zero exit, not a timeout signal.
    sb = _make_sandbox(lambda command, **kw: SimpleNamespace(result="", exit_code=124))
    with pytest.raises(RuntimeError) as ei:
        sb.exec("false")
    assert not isinstance(ei.value, SandboxCommandTimeout)


def test_exec_nonzero_raises_runtime_error():
    sb = _make_sandbox(lambda command, **kw: SimpleNamespace(result="boom", exit_code=2))
    with pytest.raises(RuntimeError):
        sb.exec("false")


def test_exec_sdk_timeout_exception_maps_to_command_timeout():
    def fake_exec(command, **kw):
        raise RuntimeError("request timed out")

    sb = _make_sandbox(fake_exec)
    with pytest.raises(SandboxCommandTimeout):
        sb.exec("sleep 999", timeout=5)


def test_exec_applies_set_env_and_user():
    seen = {}

    def fake_exec(command, **kw):
        seen["command"] = command
        return SimpleNamespace(result="", exit_code=0)

    sb = _make_sandbox(fake_exec)
    sb.set_env({"TOKEN": "abc"})
    sb.exec("run.sh", user="agent")
    # env exported inside the su-switched shell
    assert "su agent -s /bin/bash -c" in seen["command"]
    assert "export TOKEN=abc" in seen["command"]


# ---------------------------------------------------------------------------
# create retry loop (P1-4)
# ---------------------------------------------------------------------------


def _make_sandbox_with_client(create_impl):
    sb = DaytonaSandbox.__new__(DaytonaSandbox)
    sb.name = "test"
    sb._create_timeout = 120.0
    sb._client = SimpleNamespace(create=create_impl)
    return sb


def test_create_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setattr(daytona_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky_create(params, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 Too Many Requests")
        return SimpleNamespace(id="sb-123")

    sb = _make_sandbox_with_client(flaky_create)
    result = sb._create(object())
    assert result.id == "sb-123"
    assert calls["n"] == 3


def test_create_does_not_retry_non_transient(monkeypatch):
    monkeypatch.setattr(daytona_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def failing_create(params, timeout):
        calls["n"] += 1
        raise RuntimeError("Snapshot foo not found")

    sb = _make_sandbox_with_client(failing_create)
    with pytest.raises(RuntimeError, match="not found"):
        sb._create(object())
    assert calls["n"] == 1  # no retries on a non-transient error


# ---------------------------------------------------------------------------
# ephemeral default reaches the create params (no real SDK needed)
# ---------------------------------------------------------------------------


def _install_fake_daytona(monkeypatch):
    """Inject a fake ``daytona`` module whose param classes record their kwargs."""
    import types

    captured: dict = {}

    class _Params:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            captured["kwargs"] = kw

    fake = types.ModuleType("daytona")
    fake.CreateSandboxFromImageParams = _Params
    fake.CreateSandboxFromSnapshotParams = _Params
    fake.Image = type("Image", (), {})
    fake.Resources = lambda **kw: types.SimpleNamespace(**kw)
    fake.DaytonaNotFoundError = type("DaytonaNotFoundError", (Exception,), {})
    fake.DaytonaValidationError = type("DaytonaValidationError", (Exception,), {})

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def create(self, params, timeout=None):
            return SimpleNamespace(id="sb-1", delete=lambda: None, stop=lambda: None)

    fake.Daytona = _FakeClient
    monkeypatch.setitem(sys.modules, "daytona", fake)
    return captured


def test_ephemeral_defaults_passed_to_create(monkeypatch):
    """Sandboxes default to ephemeral with immediate auto-delete on stop."""
    captured = _install_fake_daytona(monkeypatch)
    DaytonaSandbox(name="t", image="python:3.11-slim")
    kw = captured["kwargs"]
    assert kw["ephemeral"] is True
    assert kw["auto_delete_interval"] == 0


def test_ephemeral_opt_out_sets_never_delete(monkeypatch):
    """Opting out of ephemeral falls back to never-auto-delete (-1)."""
    captured = _install_fake_daytona(monkeypatch)
    DaytonaSandbox(name="t", image="python:3.11-slim", ephemeral=False)
    kw = captured["kwargs"]
    assert kw["ephemeral"] is False
    assert kw["auto_delete_interval"] == -1

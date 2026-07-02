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


# ---------------------------------------------------------------------------
# silent-drop regression: long execs must ride sessions, not one idle long-poll
#
# Background (2026-07-01): `process.exec` carries the whole command on a single
# HTTP request that stays byte-silent until completion. NAT middleboxes (e.g.
# WSL2's) silently drop flows idle >~245-250s, black-holing the response — the
# command finishes but exec() blocks until the SDK read timeout and gets
# mislabeled as a wall-clock timeout. Long execs therefore MUST use the
# session API (async submit + short polls). These tests pin that contract.
# ---------------------------------------------------------------------------


class _FakeSessionProcess:
    """Fake `sandbox.process` with the session API; one-shot exec is a trap.

    ``polls_until_done`` controls how many get_session_command calls return
    "still running" before exit_code appears. ``poll_errors`` injects that
    many transient failures first.
    """

    def __init__(self, exit_code=0, stdout="session-out", stderr="", polls_until_done=2, poll_errors=0):
        self.exit_code_final = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.polls_until_done = polls_until_done
        self.poll_errors = poll_errors
        self.polls = 0
        self.one_shot_calls = 0
        self.created_sessions: list[str] = []
        self.deleted_sessions: list[str] = []
        self.submitted: list[object] = []

    # -- one-shot endpoint: must NOT be used for long commands ----------------
    def exec(self, command, **kwargs):
        self.one_shot_calls += 1
        return SimpleNamespace(result="one-shot-out", exit_code=0)

    # -- session API -----------------------------------------------------------
    def create_session(self, session_id):
        self.created_sessions.append(session_id)

    def execute_session_command(self, session_id, req):
        self.submitted.append(req)
        return SimpleNamespace(cmd_id="cmd-1")

    def get_session_command(self, session_id, cmd_id):
        if self.poll_errors > 0:
            self.poll_errors -= 1
            raise RuntimeError("transient API blip")
        self.polls += 1
        if self.polls >= self.polls_until_done:
            return SimpleNamespace(exit_code=self.exit_code_final)
        return SimpleNamespace(exit_code=None)

    def get_session_command_logs(self, session_id, cmd_id):
        return SimpleNamespace(stdout=self.stdout, stderr=self.stderr)

    def delete_session(self, session_id):
        self.deleted_sessions.append(session_id)


def _make_session_sandbox(monkeypatch, **proc_kwargs):
    pytest.importorskip("daytona")  # _exec_session imports SessionExecuteRequest lazily
    proc = _FakeSessionProcess(**proc_kwargs)
    sb = DaytonaSandbox.__new__(DaytonaSandbox)
    sb.name = "test"
    sb._closed = False
    sb._persistent_env = {}
    sb._sandbox = SimpleNamespace(process=proc)
    monkeypatch.setattr(daytona_mod.time, "sleep", lambda s: None)
    return sb, proc


def test_long_exec_routes_via_session_not_one_shot(monkeypatch):
    """timeout > threshold ⇒ session API; the droppable one-shot is untouched."""
    sb, proc = _make_session_sandbox(monkeypatch)
    out = sb.exec("sleep 300; echo hi", timeout=600)
    assert out == "session-out"
    assert proc.one_shot_calls == 0
    assert proc.created_sessions and proc.deleted_sessions == proc.created_sessions
    # async submit with the shell-timeout wrapper baked into the command
    assert getattr(proc.submitted[0], "run_async", None) is True
    assert "timeout -k 10 600" in proc.submitted[0].command


def test_short_and_untimed_execs_stay_one_shot(monkeypatch):
    """timeout ≤ threshold (and None) keep the cheap one-shot path."""
    sb, proc = _make_session_sandbox(monkeypatch)
    assert sb.exec("echo hi", timeout=30) == "one-shot-out"
    assert sb.exec("echo hi") == "one-shot-out"
    assert proc.one_shot_calls == 2
    assert proc.created_sessions == []


def test_session_exec_returns_completion_after_polls(monkeypatch):
    """The regression itself: a long exec RETURNS its output — never hangs."""
    sb, proc = _make_session_sandbox(monkeypatch, stdout="DONE\n", polls_until_done=5)
    assert sb.exec("long-job", timeout=300) == "DONE\n"
    assert proc.polls == 5


def test_session_exec_budget_kill_raises_command_timeout(monkeypatch):
    """exit 124 from the in-sandbox `timeout` still maps to SandboxCommandTimeout."""
    sb, _ = _make_session_sandbox(monkeypatch, exit_code=124)
    with pytest.raises(SandboxCommandTimeout):
        sb.exec("sleep 999", timeout=200)


def test_session_exec_nonzero_raises_runtime_error(monkeypatch):
    sb, _ = _make_session_sandbox(monkeypatch, exit_code=3, stdout="boom")
    with pytest.raises(RuntimeError) as ei:
        sb.exec("bad-job", timeout=300)
    assert not isinstance(ei.value, SandboxCommandTimeout)


def test_session_exec_tolerates_transient_poll_failures(monkeypatch):
    """A few flaky polls must not kill a long rollout."""
    sb, proc = _make_session_sandbox(monkeypatch, poll_errors=3, polls_until_done=1)
    assert sb.exec("long-job", timeout=300) == "session-out"
    assert proc.deleted_sessions  # cleanup still ran


def test_session_exec_persistent_poll_failures_raise(monkeypatch):
    sb, proc = _make_session_sandbox(monkeypatch, poll_errors=99)
    with pytest.raises(RuntimeError, match="transient API blip"):
        sb.exec("long-job", timeout=300)
    assert proc.deleted_sessions  # cleanup even on failure


def test_session_exec_poll_deadline_raises_command_timeout(monkeypatch):
    """A daemon that lost the command entirely hits the poll deadline."""
    sb, proc = _make_session_sandbox(monkeypatch, polls_until_done=10**9)
    clock = {"t": 0.0}
    monkeypatch.setattr(daytona_mod.time, "monotonic", lambda: clock["t"])

    def advance(_s):
        clock["t"] += 30.0

    monkeypatch.setattr(daytona_mod.time, "sleep", advance)
    with pytest.raises(SandboxCommandTimeout, match="poll deadline"):
        sb.exec("lost-job", timeout=200)
    assert proc.deleted_sessions

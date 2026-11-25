"""Tests for the Runta sandbox backend wrapper.

Everything here runs against a fake ``runta`` module — no SDK install, no
``RUNTA_TOKEN``, no network. The live round trip is at the bottom and is
skipped unless ``RUNTA_TOKEN`` is set.
"""

from __future__ import annotations

import builtins
import logging
import os
import shutil
import subprocess
import sys
import time
import types

import pytest

from rllm.sandbox.protocol import Sandbox, SnapshotNotFound

# ---------------------------------------------------------------------------
# Fake SDK
# ---------------------------------------------------------------------------


class FakeApiError(Exception):
    def __init__(self, message: str = "", *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class FakeResult:
    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = "", stdout_truncated: bool = False, stderr_truncated: bool = False):
        self.exit_code = exit_code
        self.stdout = stdout.encode()
        self.stdout_text = stdout
        self.stderr = stderr.encode()
        self.stderr_text = stderr
        self.stdout_truncated = stdout_truncated
        self.stderr_truncated = stderr_truncated


class FakeRuntime:
    def __init__(self, client, name, kwargs):
        self.id = f"rt-{123 + len(client.runtimes_created)}"
        self.name = name
        self.create_kwargs = kwargs
        self.execs: list[tuple[object, float | None]] = []
        self.output_caps: list[int | None] = []
        self.renewals: list[int | None] = []
        self.uploads: list[tuple[str, str]] = []
        self.deleted = 0
        self.status = "running"
        self.ready_timeouts: list[float] = []
        self._client = client
        self.files = types.SimpleNamespace(upload=lambda src, dst: self.uploads.append((src, dst)))
        self.checkpoints = types.SimpleNamespace(create=self._checkpoint)

    def wait_until_ready(self, timeout=240.0):
        self.ready_timeouts.append(timeout)
        if self._client.ready_error is not None:
            raise self._client.ready_error

    def _checkpoint(self, name=None, wait=True, timeout=240.0):
        # Like the SDK: the checkpoint exists server-side before the wait, and
        # a failed wait raises without handing back its id.
        self._client.checkpoint_names.append(name)
        if wait:
            self._client._wait_checkpoint("ckpt-1", timeout=timeout)
        return types.SimpleNamespace(id="ckpt-1", state="creating")

    def exec(self, command, timeout=None, max_output_bytes=None):
        self.execs.append((command, timeout))
        self.output_caps.append(max_output_bytes)
        if self._client.on_exec is not None:
            self._client.on_exec(command)
        return self._client.exec_results.pop(0) if self._client.exec_results else FakeResult()

    def _update_timeout_policies(self, *, auto_shutdown_after_secs=None, auto_delete_after_secs=None):
        # Reached only through a handle from ``runtimes.get``: the handle
        # ``runtimes.create`` returned belongs to the thread running the sandbox.
        if self._client.on_renew is not None:
            self._client.on_renew(self)
        if self._client.renew_error is not None:
            raise self._client.renew_error
        self.renewals.append(auto_shutdown_after_secs)

    def refresh_info(self):
        if isinstance(self.status, Exception):
            raise self.status
        return types.SimpleNamespace(status=self.status)

    def delete(self):
        if self._client.on_delete is not None:
            self._client.on_delete(self)
        self.deleted += 1


class FakeClient:
    """Stands in for ``runta.Runta``; one instance is shared by every ``Runta()`` call."""

    def __init__(self):
        self.runtimes_created: list[FakeRuntime] = []
        self.exec_results: list[FakeResult] = []
        self.create_error: Exception | None = None
        self.ready_error: Exception | None = None
        self.checkpoint_names: list[str | None] = []
        self.checkpoint_state: object = "ready"
        self.checkpoint_error: Exception | None = None
        self.checkpoint_wait_error: Exception | None = None
        self.checkpoint_waits: list[tuple[str, float]] = []
        self.deleted_checkpoints: list[str] = []
        self.closed = 0
        self.endpoints: list[str | None] = []
        self.client_options: list[dict] = []
        self.now = 1000.0  # the backend's monotonic clock
        self.lease_thread_starts = 0
        self.gets: list[str] = []
        self.renew_error: Exception | None = None
        self.on_renew = None
        self.on_exec = None
        self.on_delete = None
        self.runtimes = types.SimpleNamespace(create=self._create, get=self._get)
        self.checkpoints = types.SimpleNamespace(get=self._get_checkpoint, delete=self._delete_checkpoint, wait_until_ready=self._wait_checkpoint)

    def _create(self, name, wait=True, timeout=240.0, **kwargs):
        if self.create_error is not None:
            raise self.create_error
        # Like the SDK: the runtime exists server-side before the wait, and a
        # failed wait raises without handing back the handle.
        runtime = FakeRuntime(self, name, kwargs)
        self.runtimes_created.append(runtime)
        if wait:
            runtime.wait_until_ready(timeout=timeout)
        return runtime

    def _get(self, runtime_id):
        self.gets.append(runtime_id)
        for runtime in self.runtimes_created:
            if runtime.id == runtime_id:
                return types.SimpleNamespace(id=runtime_id, update_timeout_policies=runtime._update_timeout_policies)
        raise FakeApiError("runtime not found", status_code=404)

    def _wait_checkpoint(self, checkpoint_id, timeout=240.0):
        self.checkpoint_waits.append((checkpoint_id, timeout))
        if self.checkpoint_wait_error is not None:
            raise self.checkpoint_wait_error
        return types.SimpleNamespace(id=checkpoint_id, state="ready")

    def _get_checkpoint(self, checkpoint_id):
        if self.checkpoint_error is not None:
            raise self.checkpoint_error
        return types.SimpleNamespace(id=checkpoint_id, state=self.checkpoint_state)

    def _delete_checkpoint(self, checkpoint_id):
        if self.checkpoint_error is not None:
            raise self.checkpoint_error
        self.deleted_checkpoints.append(checkpoint_id)

    def close(self):
        self.closed += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def fake_runta(monkeypatch):
    from rllm.sandbox.backends import runta as backend

    client = FakeClient()

    def _factory(endpoint=None, **options):
        client.endpoints.append(endpoint)
        client.client_options.append(options)
        return client

    module = types.ModuleType("runta")
    module.Runta = _factory
    module.ApiError = FakeApiError
    monkeypatch.setitem(sys.modules, "runta", module)

    # Lease tests drive ``_lease_beat`` themselves on a fake clock, so the real
    # heartbeat thread stays out of the way and no test waits for an interval.
    client.start_lease_thread = backend._start_lease_thread
    monkeypatch.setattr(backend, "_start_lease_thread", lambda: setattr(client, "lease_thread_starts", client.lease_thread_starts + 1))
    monkeypatch.setattr(backend, "_monotonic", lambda: client.now)
    monkeypatch.setattr(backend, "_LEASE_CLIENTS", {})
    with backend._LIVE_LOCK:
        backend._LIVE_SANDBOXES.clear()
    return client


def _sandbox(**kwargs):
    from rllm.sandbox.backends.runta import RuntaSandbox

    return RuntaSandbox(name="t1", **kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_missing_runta_sdk_raises_friendly_install_hint(monkeypatch):
    monkeypatch.delitem(sys.modules, "runta", raising=False)
    real_import = builtins.__import__

    def _block_runta(name, *args, **kwargs):
        if name == "runta" or name.startswith("runta."):
            raise ModuleNotFoundError("No module named 'runta'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_runta)

    with pytest.raises(ImportError, match="pip install runta-sdk"):
        _sandbox()


def test_create_sandbox_factory_routes_runta_and_satisfies_protocol(fake_runta):
    from rllm.sandbox.sandboxed_flow import create_sandbox

    sb = create_sandbox("runta", name="t1", image="python:3.11-slim")
    try:
        assert isinstance(sb, Sandbox)
    finally:
        sb.close()


def test_image_runs_as_container_inside_runtime_with_lease_backstop(fake_runta):
    sb = _sandbox(image="ghcr.io/acme/task:1", vcpus=4, memory_mib=8192, timeout=3600, create_timeout=900)
    runtime = fake_runta.runtimes_created[0]

    assert runtime.name == "t1"
    assert runtime.create_kwargs == {
        "auto_shutdown_after_secs": 600,  # one lease, not the 3600s lifetime
        "auto_delete_after_secs": 600,
        "vcpus": 4,
        "memory_mib": 8192,
    }
    assert runtime.ready_timeouts == [900.0]
    command, timeout = runtime.execs[0]
    assert command == "docker run -d --name rllm-sandbox --network host ghcr.io/acme/task:1 sleep infinity"
    assert timeout == 900.0
    sb.close()


def test_failed_image_start_deletes_runtime(fake_runta):
    fake_runta.exec_results = [FakeResult(125, stderr="pull access denied")]

    with pytest.raises(RuntimeError, match="pull access denied"):
        _sandbox(image="nope:latest")

    assert fake_runta.runtimes_created[0].deleted == 1
    assert fake_runta.closed == 1


def test_runtime_that_never_becomes_ready_is_deleted(fake_runta):
    fake_runta.ready_error = TimeoutError("runtime rt-123 did not become ready within 600.0s")

    with pytest.raises(TimeoutError):
        _sandbox()

    assert fake_runta.runtimes_created[0].deleted == 1  # not left running until auto-shutdown
    assert fake_runta.closed == 1


def test_endpoint_kwarg_reaches_client(fake_runta):
    _sandbox(endpoint="https://api.example.test").close()
    assert fake_runta.endpoints == ["https://api.example.test"]


# ---------------------------------------------------------------------------
# exec
# ---------------------------------------------------------------------------


def test_exec_routes_through_docker_exec_with_user_and_timeout(fake_runta):
    sb = _sandbox()
    runtime = fake_runta.runtimes_created[0]
    fake_runta.exec_results = [FakeResult(0, stdout="hi\n")]

    assert sb.exec("echo hi && ls", timeout=30, user="agent") == "hi\n"

    argv, timeout = runtime.execs[-1]
    assert argv[:2] == ["docker", "exec"]
    assert argv[-9:-4] == ["-u", "agent", "rllm-sandbox", "bash", "-c"]  # then the tracking wrapper
    assert argv[-2].startswith("/tmp/rllm-exec-") and argv[-1] == "echo hi && ls"
    assert "SSL_CERT_FILE=/run/runta/ca-bundle.pem" in argv
    assert timeout == 30

    sb.exec("pwd")  # no timeout: nothing to kill later, so no wrapper
    assert runtime.execs[-1][0][-4:] == ["rllm-sandbox", "bash", "-c", "pwd"]
    sb.close()


def test_exec_timeout_kills_the_command_inside_the_container(fake_runta):
    sb = _sandbox()
    runtime = fake_runta.runtimes_created[0]
    fake_runta.exec_results = [FakeResult(124, stderr="Command timed out after 5 seconds.")]

    with pytest.raises(RuntimeError, match="exit 124"):
        sb.exec("sleep 600", timeout=5, user="agent")

    (timed_out, _), (kill, _) = runtime.execs[-2:]
    assert kill[:5] == ["docker", "exec", "-u", "0", "rllm-sandbox"]
    assert kill[-1] == timed_out[-2]  # the timed-out command's PID file

    # A command that finishes in time, or exits 124 on its own, triggers no kill.
    fake_runta.exec_results = [FakeResult(0), FakeResult(124)]
    count = len(runtime.execs)
    sb.exec("true", timeout=5)
    with pytest.raises(RuntimeError):
        sb.exec("exit 124")
    assert len(runtime.execs) == count + 2
    sb.close()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_tracking_wrapper_and_kill_script_stop_the_process_group(tmp_path):
    from rllm.sandbox.backends.runta import _KILL, _TRACK

    pidfile = tmp_path / "exec.pid"
    # start_new_session: a ``docker exec`` process leads its own session too.
    done = subprocess.run(["bash", "-c", _TRACK, "rllm-exec", str(pidfile), "echo out; exit 7"], capture_output=True, text=True, start_new_session=True)
    assert (done.returncode, done.stdout) == (7, "out\n")
    assert not pidfile.exists()  # in time: same result, nothing left behind

    proc = subprocess.Popen(["bash", "-c", _TRACK, "rllm-exec", str(pidfile), "sleep 300 & wait"], stdout=subprocess.PIPE, start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not (pidfile.exists() and pidfile.read_text().strip()) and time.monotonic() < deadline:
            time.sleep(0.05)
        subprocess.run(["bash", "-c", _KILL, "rllm-kill", str(pidfile)], check=True)
        proc.communicate(timeout=10)  # EOF only once every process holding stdout is dead
    finally:
        proc.kill()
    assert not pidfile.exists()


def test_exec_asks_for_a_64_mib_capture(fake_runta):
    sb = _sandbox()
    runtime = fake_runta.runtimes_created[0]

    sb.exec("cat big.log", timeout=30)
    sb.exec("cat big.log")

    assert runtime.output_caps[-2:] == [64 * 1024 * 1024] * 2  # the SDK default is 4 MiB
    sb.close()


def test_output_truncated_despite_the_larger_capture_is_reported(fake_runta, caplog):
    sb = _sandbox()
    fake_runta.exec_results = [FakeResult(0, stdout="head", stdout_truncated=True), FakeResult(0, stdout="all"), FakeResult(0, stdout="out", stderr="errhead", stderr_truncated=True)]

    with caplog.at_level(logging.WARNING, logger="rllm.sandbox.backends.runta"):
        assert sb.exec("cat big.log") == "head"
        assert sb.exec("cat small.log") == "all"
        assert sb.exec("make -j 2>&1 >/dev/null") == "out"

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert warnings[0].startswith("stdout") and "truncated" in warnings[0] and "cat big.log" in warnings[0]
    assert warnings[1].startswith("stderr") and "truncated to its first 7 bytes" in warnings[1] and "make -j" in warnings[1]
    sb.close()


def test_exec_nonzero_exit_raises_runtime_error_with_stderr_tail(fake_runta):
    sb = _sandbox()
    fake_runta.exec_results = [FakeResult(7, stdout="out", stderr="boom")]

    with pytest.raises(RuntimeError, match=r"exit 7.*\n.*\nboom"):
        sb.exec("false")
    sb.close()


def test_direct_mode_runs_on_runtime_without_container(fake_runta):
    sb = _sandbox(image=None)
    runtime = fake_runta.runtimes_created[0]
    assert runtime.execs == []  # no docker run

    sb.exec("uname -a", user="agent")
    assert runtime.execs[-1][0] == ["bash", "-c", "uname -a"]
    sb.close()


# ---------------------------------------------------------------------------
# uploads
# ---------------------------------------------------------------------------


def test_upload_file_stages_then_docker_cp(fake_runta, tmp_path):
    local = tmp_path / "patch.diff"
    local.write_text("x")
    sb = _sandbox()
    runtime = fake_runta.runtimes_created[0]

    sb.upload_file(str(local), "/work space/patch.diff")

    src, staged = runtime.uploads[0]
    assert src == str(local) and staged.startswith("/tmp/rllm-upload-")
    mkdir_argv = runtime.execs[-2][0]
    assert mkdir_argv[-1] == "mkdir -p '/work space'"
    copy = runtime.execs[-1][0]
    assert copy == f"docker cp {staged} 'rllm-sandbox:/work space/patch.diff'; rc=$?; rm -rf {staged}; exit $rc"
    sb.close()


def test_upload_dir_merges_contents_into_destination(fake_runta, tmp_path):
    (tmp_path / "tests").mkdir()
    sb = _sandbox()
    runtime = fake_runta.runtimes_created[0]

    sb.upload_dir(str(tmp_path / "tests"), "/tests/")

    staged = runtime.uploads[0][1]
    assert runtime.execs[-2][0][-1] == "mkdir -p /tests"
    assert runtime.execs[-1][0] == f"docker cp {staged}/. rllm-sandbox:/tests; rc=$?; rm -rf {staged}; exit $rc"
    sb.close()


def test_upload_dir_missing_local_path_raises(fake_runta, tmp_path):
    sb = _sandbox()
    with pytest.raises(FileNotFoundError):
        sb.upload_dir(str(tmp_path / "absent"), "/tests")
    sb.close()


def test_failed_copy_raises(fake_runta, tmp_path):
    local = tmp_path / "f"
    local.write_text("x")
    sb = _sandbox()
    fake_runta.exec_results = [FakeResult(), FakeResult(1, stderr="no such container")]

    with pytest.raises(RuntimeError, match="no such container"):
        sb.upload_file(str(local), "/a/f")
    sb.close()


# ---------------------------------------------------------------------------
# liveness / close
# ---------------------------------------------------------------------------


def test_is_alive_never_raises_and_tracks_status(fake_runta):
    sb = _sandbox()
    runtime = fake_runta.runtimes_created[0]

    assert sb.is_alive() is True
    runtime.status = types.SimpleNamespace(value="shutdown")  # enum-like
    assert sb.is_alive() is False
    runtime.status = RuntimeError("api down")
    assert sb.is_alive() is False

    sb.close()
    assert sb.is_alive() is False


def test_close_is_idempotent_and_survives_delete_failure(fake_runta):
    sb = _sandbox()
    runtime = fake_runta.runtimes_created[0]

    def _boom():
        raise FakeApiError("nope", status_code=500)

    runtime.delete = _boom
    sb.close()  # must not raise
    sb.close()
    assert fake_runta.closed == 1


# ---------------------------------------------------------------------------
# lease
# ---------------------------------------------------------------------------


def _beat():
    from rllm.sandbox.backends.runta import _lease_beat

    return _lease_beat()


def test_lease_is_renewed_every_third_of_it_up_to_the_lifetime(fake_runta):
    sb = _sandbox(timeout=1500)  # default lease: 600s
    runtime = fake_runta.runtimes_created[0]
    assert runtime.create_kwargs["auto_shutdown_after_secs"] == 600
    assert fake_runta.lease_thread_starts == 1

    fake_runta.now += 100
    assert _beat() == 100 and runtime.renewals == []  # not due before lease / 3

    fake_runta.now += 100
    for _ in range(4):  # t = 200, 400, 600, 800
        assert _beat() == 200
        fake_runta.now += 200
    assert runtime.renewals == [600] * 4

    assert _beat() is None  # t = 1000: 500s of lifetime left
    assert runtime.renewals[-1] == 500  # the clock now ends with the lifetime

    fake_runta.now += 5000
    assert _beat() is None and len(runtime.renewals) == 5  # never past the cap
    sb.close()


def test_lease_and_lifetime_bound_the_create_time_clock(fake_runta):
    sb = _sandbox(timeout=3600, lease=120)
    short = _sandbox(timeout=300)  # lifetime within one lease: the clock already ends there
    by_kwarg, by_lifetime = fake_runta.runtimes_created

    assert by_kwarg.create_kwargs["auto_shutdown_after_secs"] == 120
    assert by_lifetime.create_kwargs["auto_shutdown_after_secs"] == 300
    assert fake_runta.lease_thread_starts == 1  # nothing to renew for the short one

    fake_runta.now += 40
    assert _beat() == 40
    assert (by_kwarg.renewals, by_lifetime.renewals) == ([120], [])

    with pytest.raises(ValueError, match="lease"):
        _sandbox(lease=0)  # 0 would switch Runta's shutdown clock off
    sb.close()
    short.close()


def test_failed_renewal_is_logged_and_retried_without_reaching_the_caller(fake_runta, caplog):
    sb = _sandbox(timeout=1500)
    runtime = fake_runta.runtimes_created[0]
    fake_runta.renew_error = FakeApiError("runtime revision conflict: expected 3", status_code=409)

    fake_runta.now += 200
    with caplog.at_level(logging.WARNING, logger="rllm.sandbox.backends.runta"):
        assert _beat() == 200  # no exception; due again on the next beat
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "renew" in warnings[0] and "revision conflict" in warnings[0]
    assert runtime.renewals == []

    fake_runta.exec_results = [FakeResult(0, stdout="ok")]
    assert sb.exec("true") == "ok"  # the sandbox is unaffected

    fake_runta.renew_error = None
    fake_runta.now += 200
    _beat()
    assert runtime.renewals == [600]

    fake_runta.renew_error = RuntimeError("connection reset")
    fake_runta.now += 200
    _beat()
    sb.close()  # must not raise
    assert runtime.deleted == 1


def test_renewal_stops_once_the_lifetime_is_used_up(fake_runta):
    sb = _sandbox(timeout=900)
    runtime = fake_runta.runtimes_created[0]

    fake_runta.renew_error = TimeoutError("api down")
    for _ in range(4):  # t = 200 .. 800: every renewal fails, each is retried
        fake_runta.now += 200
        assert _beat() == 200
    fake_runta.renew_error = None
    fake_runta.now += 200  # t = 1000: past the 900s lifetime
    assert _beat() is None and runtime.renewals == []
    sb.close()


def test_closed_sandbox_is_not_renewed(fake_runta):
    sb = _sandbox(timeout=3600)
    other = _sandbox(timeout=3600)
    closed, live = fake_runta.runtimes_created

    sb.close()
    fake_runta.now += 200
    assert _beat() == 200
    assert (closed.renewals, live.renewals) == ([], [600])

    other.close()
    fake_runta.now += 200
    assert _beat() is None and live.renewals == [600]


def test_heartbeat_has_its_own_client_and_holds_no_registry_lock_during_a_renewal(fake_runta):
    from rllm.sandbox.backends import runta as backend

    sb = _sandbox(timeout=3600, endpoint="https://api.example.test")
    seen = []

    def _on_renew(runtime):
        free = backend._LIVE_LOCK.acquire(blocking=False)
        if free:
            backend._LIVE_LOCK.release()
        seen.append((free, sb._lease_lock.locked()))

    fake_runta.on_renew = _on_renew
    fake_runta.on_delete = lambda runtime: seen.append(("delete", sb._lease_lock.locked()))
    fake_runta.now += 200
    _beat()
    fake_runta.now += 200
    _beat()

    # A second client for the same endpoint, built once and used for every beat;
    # the runtime handle comes from it, not from the sandbox.
    assert fake_runta.endpoints == ["https://api.example.test"] * 2
    assert fake_runta.client_options == [{}, {"retry_max_attempts": 1}]
    assert fake_runta.gets == ["rt-123"]
    sb.close()
    # Registry lock free during the call; a renewal and delete() exclude each
    # other, because a renewal changes the revision delete() relies on.
    assert seen == [(True, True), (True, True), ("delete", True)]


def test_heartbeat_thread_is_one_daemon_for_the_module(fake_runta, monkeypatch):
    from rllm.sandbox.backends import runta as backend

    monkeypatch.setattr(backend, "_lease_beat", lambda: None)  # nothing due: the thread just waits
    fake_runta.start_lease_thread()
    thread = backend._LEASE_THREAD
    fake_runta.start_lease_thread()

    assert backend._LEASE_THREAD is thread
    assert thread.is_alive() and thread.daemon  # never keeps the interpreter alive


# ---------------------------------------------------------------------------
# snapshots (checkpoints)
# ---------------------------------------------------------------------------


def test_snapshot_ref_restores_checkpoint_without_pulling(fake_runta):
    sb = _sandbox(image="runta-checkpoint:ckpt-9")
    runtime = fake_runta.runtimes_created[0]

    assert runtime.create_kwargs["checkpoint_id"] == "ckpt-9"
    assert [cmd for cmd, _ in runtime.execs] == ["docker start rllm-sandbox"]

    sb.exec("true")
    assert runtime.execs[-1][0][:2] == ["docker", "exec"]
    sb.close()


def test_missing_checkpoint_raises_snapshot_not_found(fake_runta):
    fake_runta.create_error = FakeApiError("checkpoint not found", status_code=404)
    with pytest.raises(SnapshotNotFound):
        _sandbox(image="runta-checkpoint:gone")
    assert fake_runta.closed == 1


def test_not_found_while_waiting_for_restore_is_not_a_missing_snapshot(fake_runta):
    fake_runta.ready_error = FakeApiError("runtime not found", status_code=404)

    with pytest.raises(FakeApiError):  # the checkpoint existed; get_sandbox must not fall back to cold
        _sandbox(image="runta-checkpoint:ckpt-9")

    assert fake_runta.runtimes_created[0].deleted == 1


def test_other_create_errors_propagate(fake_runta):
    fake_runta.create_error = FakeApiError("quota", status_code=429)
    with pytest.raises(FakeApiError):
        _sandbox(image="runta-checkpoint:x")

    fake_runta.create_error = FakeApiError("not found", status_code=404)
    with pytest.raises(FakeApiError):  # 404 on a cold create is a real error, not a missing snapshot
        _sandbox(image="python:3.11-slim")


def test_checkpoint_returns_prefixed_ref(fake_runta):
    sb = _sandbox()
    assert sb.checkpoint("rllm-env-abc") == "runta-checkpoint:ckpt-1"
    assert fake_runta.checkpoint_names == ["rllm-env-abc"]
    assert fake_runta.checkpoint_waits == [("ckpt-1", 600.0)]
    assert fake_runta.deleted_checkpoints == []
    sb.close()


@pytest.mark.parametrize("error", [TimeoutError("checkpoint ckpt-1 did not become ready"), RuntimeError("checkpoint ckpt-1 entered failed state")])
def test_checkpoint_that_does_not_become_ready_is_deleted(fake_runta, error):
    sb = _sandbox()
    fake_runta.checkpoint_wait_error = error

    with pytest.raises(type(error)):
        sb.checkpoint("rllm-env-abc")
    assert fake_runta.deleted_checkpoints == ["ckpt-1"]  # no untracked leftover per retry

    fake_runta.checkpoint_error = FakeApiError("conflict", status_code=409)  # delete fails too
    with pytest.raises(type(error)):  # still the original failure
        sb.checkpoint("rllm-env-abc")
    sb.close()


def test_snapshot_absent_only_on_confirmed_gone(fake_runta):
    from rllm.sandbox.sandboxed_flow import snapshot_absent

    assert snapshot_absent("runta", "runta-checkpoint:c") is False  # ready → present
    fake_runta.checkpoint_state = types.SimpleNamespace(value="failed")
    assert snapshot_absent("runta", "runta-checkpoint:c") is True
    fake_runta.checkpoint_error = FakeApiError("not found", status_code=404)
    assert snapshot_absent("runta", "runta-checkpoint:c") is True
    fake_runta.checkpoint_error = FakeApiError("unauthorized", status_code=401)
    assert snapshot_absent("runta", "runta-checkpoint:c") is False  # cannot confirm → keep


def test_delete_snapshot_reports_confirmed_deletion(fake_runta):
    from rllm.sandbox.sandboxed_flow import delete_snapshot

    assert delete_snapshot("runta", "runta-checkpoint:c") is True
    assert fake_runta.deleted_checkpoints == ["c"]
    fake_runta.checkpoint_error = FakeApiError("not found", status_code=404)
    assert delete_snapshot("runta", "runta-checkpoint:c") is True  # already gone
    fake_runta.checkpoint_error = FakeApiError("forbidden", status_code=403)
    assert delete_snapshot("runta", "runta-checkpoint:c") is False


def test_build_snapshot_reuses_live_prior_ref(fake_runta):
    from rllm.sandbox.sandboxed_flow import build_snapshot

    ref = build_snapshot("runta", task=None, key="k", prior_ref="runta-checkpoint:old")
    assert ref == "runta-checkpoint:old"
    assert fake_runta.runtimes_created == []  # no build runtime


def test_long_snapshot_build_keeps_its_runtime_through_the_lease(fake_runta, tmp_path):
    from rllm.sandbox.sandboxed_flow import build_snapshot

    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "Dockerfile").write_text("FROM python:3.11-slim\nRUN make deps\nRUN make build\nRUN make install\n")
    task = types.SimpleNamespace(id="t", task_dir=tmp_path, dataset_dir=tmp_path, metadata={})

    def _slow_step(command):
        # Each build step blocks the building thread for 800s, longer than a
        # lease; meanwhile the heartbeat thread beats on its own schedule.
        for _ in range(4):
            fake_runta.now += 200
            _beat()

    fake_runta.on_exec = _slow_step
    assert build_snapshot("runta", task=task, key="k") == "runta-checkpoint:ckpt-1"

    runtime = fake_runta.runtimes_created[0]
    assert runtime.create_kwargs["auto_shutdown_after_secs"] == 600  # not the 3300s build lifetime
    # Four steps (image start + three RUNs) took 3200s; the clock was restarted
    # every 200s and never ran out, then capped at the build lifetime.
    assert runtime.renewals == [600] * 13 + [500]
    assert fake_runta.checkpoint_names == ["k"]
    assert runtime.deleted == 1


def test_resource_kwargs_map_task_environment():
    from rllm.eval._resolution import _sandbox_resource_kwargs

    task = types.SimpleNamespace(metadata={"environment": {"cpus": 1.5, "memory_mb": 4096, "build_timeout_sec": 1800}})
    assert _sandbox_resource_kwargs(task, "runta") == {"vcpus": 2, "memory_mib": 4096, "create_timeout": 1800.0}


@pytest.mark.parametrize(("storage_mb", "disk_gib"), [(30720, 32), (10240, 32), (51200, 50), (40961, 41), (262144, 256)])
def test_resource_kwargs_map_storage_to_runta_disk(storage_mb, disk_gib):
    from rllm.eval._resolution import _sandbox_resource_kwargs

    task = types.SimpleNamespace(id="t", metadata={"environment": {"storage_mb": storage_mb}})
    assert _sandbox_resource_kwargs(task, "runta")["disk_gib"] == disk_gib  # whole GiB, rounded up, 32 at least


def test_resource_kwargs_reject_storage_above_runta_maximum():
    from rllm.eval._resolution import _sandbox_resource_kwargs

    task = types.SimpleNamespace(id="big-task", metadata={"environment": {"storage_mb": 262145}})
    with pytest.raises(ValueError, match=r"big-task.*storage_mb=262145.*256 GiB"):
        _sandbox_resource_kwargs(task, "runta")


# The only shape in which the SDK sends a disk size: ``resources`` as a mapping,
# which replaces the flat ``vcpus`` / ``memory_mib`` arguments.
_DISK_RESOURCES = {"requests": {"vcpus": 4, "memory_mib": 8192, "disk_gib": 40}, "limits": {"memory_mib": 8192}}


def test_disk_request_travels_in_resources_with_vcpus_and_memory(fake_runta):
    _sandbox(vcpus=4, memory_mib=8192, disk_gib=40).close()
    _sandbox(memory_mib=2048, memory_max_mib=4096, disk_gib=32).close()
    full, defaults = (runtime.create_kwargs for runtime in fake_runta.runtimes_created)

    assert full["resources"] == _DISK_RESOURCES
    assert defaults["resources"] == {"requests": {"vcpus": 1, "memory_mib": 2048, "disk_gib": 32}, "limits": {"memory_mib": 4096}}
    for kwargs in (full, defaults):  # the SDK ignores these once ``resources`` is given
        assert not {"vcpus", "memory_mib", "memory_max_mib", "disk_gib"} & kwargs.keys()


def test_snapshot_restore_sends_no_disk_request(fake_runta):
    _sandbox(image="runta-checkpoint:ckpt-9", vcpus=4, memory_mib=8192, disk_gib=40).close()

    # A checkpoint fixes the disk size, and a live checkpoint rejects ``resources``.
    kwargs = fake_runta.runtimes_created[0].create_kwargs
    assert kwargs["checkpoint_id"] == "ckpt-9"
    assert "resources" not in kwargs and "disk_gib" not in kwargs


def test_real_sdk_sends_the_disk_request_with_vcpus_and_memory():
    transport = pytest.importorskip("runta._internal.transport")

    body = transport.create_runtime_request(
        "t1", checkpoint_id=None, vcpus=1, memory_mib=1024, memory_max_mib=None, resources=_DISK_RESOURCES, ingress_specs=None, egress_policy=None, idle_suspend_after_secs=None
    )
    assert body.to_dict()["resources"] == _DISK_RESOURCES


# ---------------------------------------------------------------------------
# Live round trip (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("RUNTA_TOKEN"), reason="RUNTA_TOKEN not set")
def test_live_round_trip(tmp_path):
    pytest.importorskip("runta")
    from rllm.sandbox.backends.runta import RuntaSandbox

    (tmp_path / "d" / "sub").mkdir(parents=True)
    (tmp_path / "d" / "sub" / "a.txt").write_text("nested")
    (tmp_path / "f.txt").write_text("single")

    sb = RuntaSandbox(name="rllm-live-test", image="python:3.11-slim", timeout=900)
    try:
        assert sb.is_alive()
        assert sb.exec("python -c 'print(6*7)'").strip() == "42"
        sb.upload_file(str(tmp_path / "f.txt"), "/work/f.txt")
        sb.upload_dir(str(tmp_path / "d"), "/work/d")
        assert sb.exec("cat /work/f.txt /work/d/sub/a.txt") == "singlenested"
        with pytest.raises(RuntimeError, match="exit 3"):
            sb.exec("exit 3")
    finally:
        sb.close()
    assert not sb.is_alive()

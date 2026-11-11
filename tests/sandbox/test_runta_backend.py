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
    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = "", stdout_truncated: bool = False):
        self.exit_code = exit_code
        self.stdout = stdout.encode()
        self.stdout_text = stdout
        self.stderr_text = stderr
        self.stdout_truncated = stdout_truncated


class FakeRuntime:
    def __init__(self, client, name, kwargs):
        self.id = "rt-123"
        self.name = name
        self.create_kwargs = kwargs
        self.execs: list[tuple[object, float | None]] = []
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

    def exec(self, command, timeout=None):
        self.execs.append((command, timeout))
        return self._client.exec_results.pop(0) if self._client.exec_results else FakeResult()

    def refresh_info(self):
        if isinstance(self.status, Exception):
            raise self.status
        return types.SimpleNamespace(status=self.status)

    def delete(self):
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
        self.runtimes = types.SimpleNamespace(create=self._create)
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
    client = FakeClient()

    def _factory(endpoint=None):
        client.endpoints.append(endpoint)
        return client

    module = types.ModuleType("runta")
    module.Runta = _factory
    module.ApiError = FakeApiError
    monkeypatch.setitem(sys.modules, "runta", module)
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


def test_image_runs_as_container_inside_runtime_with_lifetime_backstop(fake_runta):
    sb = _sandbox(image="ghcr.io/acme/task:1", vcpus=4, memory_mib=8192, timeout=3600, create_timeout=900)
    runtime = fake_runta.runtimes_created[0]

    assert runtime.name == "t1"
    assert runtime.create_kwargs == {
        "auto_shutdown_after_secs": 3600,
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


def test_truncated_stdout_is_reported(fake_runta, caplog):
    sb = _sandbox()
    fake_runta.exec_results = [FakeResult(0, stdout="head", stdout_truncated=True), FakeResult(0, stdout="all")]

    with caplog.at_level(logging.WARNING, logger="rllm.sandbox.backends.runta"):
        assert sb.exec("cat big.log") == "head"
        assert sb.exec("cat small.log") == "all"

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "truncated" in warnings[0] and "cat big.log" in warnings[0]
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


def test_resource_kwargs_map_task_environment():
    from rllm.eval._resolution import _sandbox_resource_kwargs

    task = types.SimpleNamespace(metadata={"environment": {"cpus": 1.5, "memory_mb": 4096, "build_timeout_sec": 1800}})
    assert _sandbox_resource_kwargs(task, "runta") == {"vcpus": 2, "memory_mib": 4096, "create_timeout": 1800.0}


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

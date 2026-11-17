"""Tests for the OpenSandbox sandbox backend wrapper.

Covers behavior that doesn't require the real ``opensandbox`` SDK (or an
OPEN_SANDBOX_API_KEY): the friendly error when the SDK isn't installed, and
protocol semantics (exec, uploads, liveness, close, snapshot routing) against
a fake SDK injected into ``sys.modules``.
"""

from __future__ import annotations

import builtins
import io
import sys
import tarfile
import types
from datetime import timedelta

import pytest

from rllm.sandbox.backends.opensandbox import OpenSandboxSandbox

# --------------------------------------------------------------------------
# Fake SDK plumbing
# --------------------------------------------------------------------------


class _Recorder:
    """Collects the calls the backend makes into the fake SDK."""

    def __init__(self):
        self.create_calls: list[dict] = []
        self.run_calls: list[dict] = []
        self.written_files: dict[str, bytes] = {}
        self.snapshots: dict[str, str] = {}  # id -> state ("__404__" = raise 404)
        self.deleted: list[str] = []
        self.create_error: Exception | None = None
        self.execution: object | None = None
        self.healthy = True
        self.destroy_calls = 0


def _make_execution(exit_code=0, stdout="ok", stderr=""):
    """A stand-in for ``opensandbox.models.execd.Execution``."""

    class _Msg:
        def __init__(self, text):
            self.text = text

    class _Logs:
        def __init__(self, stdout, stderr):
            self.stdout = stdout
            self.stderr = stderr

    class _Exec:
        pass

    ex = _Exec()
    ex.logs = _Logs([_Msg(stdout)], [_Msg(stderr)] if stderr else [])
    ex.exit_code = exit_code
    return ex


def _install_fake_sdk(monkeypatch, recorder: _Recorder) -> None:
    """Inject a minimal fake ``opensandbox`` package into ``sys.modules``."""
    opensandbox = types.ModuleType("opensandbox")
    exceptions = types.ModuleType("opensandbox.exceptions")

    class SandboxApiException(Exception):
        def __init__(self, message=None, status_code=None, **kwargs):
            super().__init__(message or "api error")
            self.status_code = status_code

    class SandboxException(Exception):
        pass

    exceptions.SandboxApiException = SandboxApiException
    exceptions.SandboxException = SandboxException

    models = types.ModuleType("opensandbox.models")
    sandboxes = types.ModuleType("opensandbox.models.sandboxes")

    class SandboxImageSpec:
        def __init__(self, image=None, **kwargs):
            self.image = image

    sandboxes.SandboxImageSpec = SandboxImageSpec

    execd = types.ModuleType("opensandbox.models.execd")

    class RunCommandOpts:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    execd.RunCommandOpts = RunCommandOpts

    sync = types.ModuleType("opensandbox.sync")

    def _raise_404_if_requested(sid):
        if recorder.snapshots.get(sid) == "__404__":
            raise exceptions.SandboxApiException("Get snapshot failed: HTTP 404", status_code=404)

    class SandboxSync:
        @staticmethod
        def create(image=None, **kwargs):
            recorder.create_calls.append({"image": image, **kwargs})
            if recorder.create_error is not None:
                raise recorder.create_error

            class _Commands:
                @staticmethod
                def run(command, opts=None):
                    recorder.run_calls.append({"command": command, "opts": opts})
                    return recorder.execution

            class _Files:
                @staticmethod
                def write_file(path, data):
                    recorder.written_files[path] = data

            class _Sb:
                id = "sb-123"
                commands = _Commands()
                files = _Files()

                def is_healthy(self):
                    return recorder.healthy

                def destroy(self):
                    recorder.destroy_calls += 1

            return _Sb()

    class SandboxManagerSync:
        @staticmethod
        def create(connection_config=None):
            return _Manager(recorder, _raise_404_if_requested)

    class _Manager:
        def __init__(self, rec, raise_404):
            self._rec = rec
            self._raise_404 = raise_404

        def get_snapshot(self, snapshot_id):
            self._raise_404(snapshot_id)
            state = self._rec.snapshots.get(snapshot_id, "Ready")

            class _Status:
                pass

            class _Info:
                pass

            info = _Info()
            info.status = _Status()
            info.status.state = state
            return info

        def delete_snapshot(self, snapshot_id):
            self._raise_404(snapshot_id)
            self._rec.deleted.append(snapshot_id)

    sync.SandboxSync = SandboxSync
    sync.SandboxManagerSync = SandboxManagerSync

    manager_mod = types.ModuleType("opensandbox.sync.manager")
    manager_mod.SandboxManagerSync = SandboxManagerSync

    for name, mod in [
        ("opensandbox", opensandbox),
        ("opensandbox.exceptions", exceptions),
        ("opensandbox.models", models),
        ("opensandbox.models.sandboxes", sandboxes),
        ("opensandbox.models.execd", execd),
        ("opensandbox.sync", sync),
        ("opensandbox.sync.manager", manager_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.fixture
def fake_sdk(monkeypatch):
    recorder = _Recorder()
    _install_fake_sdk(monkeypatch, recorder)
    return recorder


def _make_sandbox(recorder, image="python:3.11-slim", **kwargs) -> OpenSandboxSandbox:
    recorder.execution = _make_execution()
    return OpenSandboxSandbox(name="test-sb", image=image, **kwargs)


# --------------------------------------------------------------------------
# Missing SDK
# --------------------------------------------------------------------------


def test_missing_opensandbox_sdk_raises_friendly_install_hint(monkeypatch):
    """When the ``opensandbox`` package isn't installed, instantiating
    OpenSandboxSandbox should raise an ImportError naming the install command,
    not a bare ``ModuleNotFoundError("No module named 'opensandbox'")``.
    """
    for name in list(sys.modules):
        if name == "opensandbox" or name.startswith("opensandbox."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    real_import = builtins.__import__

    def _block_opensandbox(name, *args, **kwargs):
        if name == "opensandbox" or name.startswith("opensandbox."):
            raise ModuleNotFoundError("No module named 'opensandbox'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_opensandbox)

    with pytest.raises(ImportError, match="pip install opensandbox"):
        OpenSandboxSandbox(name="test")


# --------------------------------------------------------------------------
# exec / upload / lifecycle semantics
# --------------------------------------------------------------------------


def test_exec_returns_stdout(fake_sdk):
    sb = _make_sandbox(fake_sdk)
    assert sb.exec("echo ok") == "ok"
    call = fake_sdk.run_calls[0]
    assert call["command"] == "echo ok"
    assert call["opts"] is not None and call["opts"].kwargs == {}


def test_exec_raises_runtime_error_on_nonzero_exit(fake_sdk):
    sb = _make_sandbox(fake_sdk)
    fake_sdk.execution = _make_execution(exit_code=2, stdout="", stderr="boom")
    with pytest.raises(RuntimeError, match="exit 2"):
        sb.exec("false")
    # A missing exit code (command never spawned) is also a failure.
    fake_sdk.execution = _make_execution(exit_code=None, stdout="", stderr="spawn lost")
    with pytest.raises(RuntimeError, match="exit None"):
        sb.exec("nope")


def test_exec_maps_timeout_and_numeric_uid_into_opts(fake_sdk):
    sb = _make_sandbox(fake_sdk)
    sb.exec("sleep 1", timeout=5.0, user="1000")
    assert fake_sdk.run_calls[0]["opts"].kwargs == {"timeout": timedelta(seconds=5.0), "uid": 1000}
    # A non-numeric username can't be resolved client-side and is ignored.
    sb.exec("whoami", user="root")
    assert fake_sdk.run_calls[1]["opts"].kwargs == {}


def test_upload_file_writes_bytes_via_files_api(fake_sdk, tmp_path):
    sb = _make_sandbox(fake_sdk)
    src = tmp_path / "patch.diff"
    src.write_bytes(b"hello")
    sb.upload_file(str(src), "/workspace/patch.diff")
    assert fake_sdk.written_files["/workspace/patch.diff"] == b"hello"
    # The parent directory is created first (best-effort).
    assert "mkdir" in fake_sdk.run_calls[0]["command"]


def test_upload_dir_tar_roundtrip(fake_sdk, tmp_path):
    sb = _make_sandbox(fake_sdk)
    src = tmp_path / "tree"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "leaf.txt").write_text("leaf")
    sb.upload_dir(str(src), "/workspace/tree")

    # One tar.gz went through the files API; it extracts to the requested name.
    assert len(fake_sdk.written_files) == 1
    remote_tar, payload = next(iter(fake_sdk.written_files.items()))
    assert remote_tar.startswith("/tmp/_upload_")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        assert "tree/sub/leaf.txt" in tar.getnames()
    extract = fake_sdk.run_calls[-1]["command"]
    assert "--no-same-owner" in extract and "/workspace" in extract


def test_upload_dir_missing_local_path_raises(fake_sdk, tmp_path):
    sb = _make_sandbox(fake_sdk)
    with pytest.raises(FileNotFoundError):
        sb.upload_dir(str(tmp_path / "nope"), "/workspace/x")


def test_is_alive_never_raises_and_respects_closed(fake_sdk):
    sb = _make_sandbox(fake_sdk)
    assert sb.is_alive() is True
    fake_sdk.healthy = False
    assert sb.is_alive() is False

    def boom():
        raise RuntimeError("ping endpoint gone")

    fake_sdk.healthy = True
    sb._sandbox.is_healthy = boom
    assert sb.is_alive() is False  # error in the check counts as dead

    sb.close()
    assert sb.is_alive() is False


def test_close_destroys_once(fake_sdk):
    sb = _make_sandbox(fake_sdk)
    sb.close()
    assert fake_sdk.destroy_calls == 1
    sb.close()  # idempotent
    assert fake_sdk.destroy_calls == 1


# --------------------------------------------------------------------------
# Image vs snapshot-id routing
# --------------------------------------------------------------------------


def test_docker_image_routes_to_image_param(fake_sdk):
    _make_sandbox(fake_sdk, image="python:3.11-slim")
    assert fake_sdk.create_calls[0]["image"] is not None
    assert "snapshot_id" not in fake_sdk.create_calls[0]


def test_snapshot_ref_routes_to_snapshot_id_param(fake_sdk):
    fake_sdk.execution = _make_execution()
    OpenSandboxSandbox(name="test-sb", image="snap-abc123")
    assert fake_sdk.create_calls[0]["image"] is None
    assert fake_sdk.create_calls[0]["snapshot_id"] == "snap-abc123"


def test_missing_snapshot_404_raises_snapshot_not_found(fake_sdk):
    from opensandbox.exceptions import SandboxApiException  # the fake

    from rllm.sandbox.protocol import SnapshotNotFound

    fake_sdk.create_error = SandboxApiException("Get snapshot failed: HTTP 404", status_code=404)
    with pytest.raises(SnapshotNotFound, match="snap-gone"):
        OpenSandboxSandbox(name="test-sb", image="snap-gone")


def test_other_api_errors_propagate(fake_sdk):
    from opensandbox.exceptions import SandboxApiException  # the fake

    fake_sdk.create_error = SandboxApiException("boom", status_code=500)
    with pytest.raises(SandboxApiException):
        OpenSandboxSandbox(name="test-sb", image="snap-x")


def test_image_ref_404_is_not_snapshot_not_found(fake_sdk):
    """A 404 on a plain image ref is a real miss — it must not be converted
    into SnapshotNotFound (the cold path would then boot the same image and
    surface a misleading exception type)."""
    from opensandbox.exceptions import SandboxApiException  # the fake

    from rllm.sandbox.protocol import SnapshotNotFound

    fake_sdk.create_error = SandboxApiException("image not found: HTTP 404", status_code=404)
    with pytest.raises(SandboxApiException) as excinfo:
        OpenSandboxSandbox(name="test-sb", image="ghcr.io/foo/gone:tag")
    assert not isinstance(excinfo.value, SnapshotNotFound)


# --------------------------------------------------------------------------
# Snapshot admin probes
# --------------------------------------------------------------------------


def test_opensandbox_ref_absent_confirmed_notfound_only(fake_sdk):
    from rllm.sandbox.backends.opensandbox import _opensandbox_ref_absent

    fake_sdk.snapshots["snap-gone"] = "__404__"  # the fake raises 404 for this id
    assert _opensandbox_ref_absent("snap-gone") is True  # confirmed gone
    assert _opensandbox_ref_absent("snap-unknown") is False  # default fake state Ready → keep
    fake_sdk.snapshots["snap-failed"] = "Failed"
    assert _opensandbox_ref_absent("snap-failed") is True  # terminal state → absent
    fake_sdk.snapshots["snap-creating"] = "Creating"
    assert _opensandbox_ref_absent("snap-creating") is False  # not terminal → keep


def test_opensandbox_ref_alive_requires_ready_state(fake_sdk):
    from rllm.sandbox.backends.opensandbox import _opensandbox_ref_alive

    fake_sdk.snapshots["snap-live"] = "Ready"
    assert _opensandbox_ref_alive("snap-live") is True
    fake_sdk.snapshots["snap-creating"] = "Creating"
    assert _opensandbox_ref_alive("snap-creating") is False  # not confirmed — rebuild
    fake_sdk.snapshots["snap-gone"] = "__404__"
    assert _opensandbox_ref_alive("snap-gone") is False


def test_delete_opensandbox_snapshot(fake_sdk):
    from rllm.sandbox.backends.opensandbox import delete_opensandbox_snapshot

    fake_sdk.snapshots["snap-404"] = "__404__"
    assert delete_opensandbox_snapshot("snap-x") is True
    assert fake_sdk.deleted == ["snap-x"]
    assert delete_opensandbox_snapshot("snap-404") is True  # already gone — safe to drop
    assert fake_sdk.deleted == ["snap-x"]  # the 404 path never reached the fake


# --------------------------------------------------------------------------
# Wiring: factory dispatch + resource kwargs
# --------------------------------------------------------------------------


def test_create_sandbox_factory_dispatches_to_opensandbox(fake_sdk):
    from rllm.sandbox.sandboxed_flow import create_sandbox

    sb = create_sandbox("opensandbox", name="wired", image="python:3.11-slim")
    assert isinstance(sb, OpenSandboxSandbox)
    assert fake_sdk.create_calls[0]["metadata"]["rllm_name"] == "wired"
    sb.close()


def test_snapshot_dispatch_wiring(fake_sdk):
    from rllm.sandbox import sandboxed_flow

    assert sandboxed_flow.snapshot_absent("opensandbox", "snap-live") is False
    assert sandboxed_flow.delete_snapshot("opensandbox", "snap-live") is True
    assert fake_sdk.deleted == ["snap-live"]
    # docker/local have no snapshots — still None/False.
    assert sandboxed_flow.build_snapshot("docker", None, "k") is None
    assert sandboxed_flow.delete_snapshot("docker", "k") is False
    assert sandboxed_flow.snapshot_absent("docker", "k") is False


def test_resource_kwargs_mapping():
    from rllm.eval._resolution import _sandbox_resource_kwargs
    from rllm.types import Task

    task = Task(id="t", instruction="", metadata={"environment": {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240}})
    kw = _sandbox_resource_kwargs(task, "opensandbox")
    assert kw["resource"] == {"cpu": "2", "memory": "4Gi"}
    assert kw["ready_timeout"] == 600.0

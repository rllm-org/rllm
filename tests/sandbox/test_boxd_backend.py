"""Tests for the boxd sandbox backend wrapper.

Runs against a fake ``boxd`` client, so neither the real SDK's network
calls nor a BOXD_API_KEY are needed. The live check is
``tests/sandbox/test_boxd_live.py`` (``BOXD_E2E=1``).
"""

from __future__ import annotations

import builtins
import io
import sys
import tarfile
import types
from dataclasses import dataclass, field

import pytest

from rllm.sandbox.protocol import Sandbox, SnapshotNotFound

# ---------------------------------------------------------------------------
# Fake SDK
# ---------------------------------------------------------------------------


class BoxdError(Exception):
    def __init__(self, message: str = "", code: str | None = None):
        super().__init__(message)
        self.message = message
        self.code = code


class NotFoundError(BoxdError):
    pass


class ConflictError(BoxdError):
    pass


class AuthenticationError(BoxdError):
    pass


class PermissionDeniedError(BoxdError):
    pass


@dataclass
class _Machine:
    id: str
    name: str
    status: str = "running"
    labels: dict = field(default_factory=dict)


@dataclass
class _Result:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


@dataclass
class _Snapshot:
    name: str
    status: str = "ready"


class _Files:
    def __init__(self, client):
        self.client = client
        self.uploads: dict[str, bytes] = {}

    def upload(self, machine_id, path, data, *, region=None):
        self.client.calls.append(("upload", machine_id, path))
        self.uploads[path] = data if isinstance(data, bytes) else data.encode()
        return len(self.uploads[path])


class _Machines:
    def __init__(self, client):
        self.client = client
        self.files = _Files(client)
        self.created: list[dict] = []
        self.by_id: dict[str, _Machine] = {}
        self.exec_results: list = []  # queued results / exceptions, else exit 0
        self.execs: list = []
        self.exec_env: list = []
        self.deleted: list[str] = []
        self.started: list[str] = []
        self.hibernated: list[str] = []
        self.delete_error: Exception | None = None
        self.create_error: Exception | None = None

    def create(self, name=None, **kw):
        if self.create_error is not None:
            raise self.create_error
        mid = f"m-{len(self.created) + 1}"
        self.created.append({"name": name, **kw})
        m = _Machine(id=mid, name=name, labels=dict(kw.get("labels") or {}))
        self.by_id[mid] = m
        return _Machine(id=mid, name=name, status="pending")

    def wait_until_ready(self, machine_id, *, timeout=90.0, region=None):
        self.client.calls.append(("wait", machine_id, timeout))
        return self.by_id[machine_id]

    def exec(self, machine_id, command, *, env=None, timeout=None, region=None, **kw):
        self.execs.append((machine_id, command, timeout))
        self.exec_env.append(env)
        if self.exec_results:
            r = self.exec_results.pop(0)
            if isinstance(r, Exception):
                raise r
            return r
        return _Result()

    def get(self, machine_id, *, region=None):
        if machine_id not in self.by_id or machine_id in self.deleted:
            raise NotFoundError("machine not found")
        return self.by_id[machine_id]

    def start(self, machine_id, *, region=None):
        self.started.append(machine_id)

    def delete(self, machine_id, *, region=None):
        if self.delete_error is not None:
            raise self.delete_error
        if machine_id in self.deleted:
            raise NotFoundError("machine not found")
        self.deleted.append(machine_id)

    def hibernate(self, machine_id, *, region=None):
        self.hibernated.append(machine_id)


class _Snapshots:
    def __init__(self, client):
        self.client = client
        self.items: dict[str, _Snapshot] = {}
        self.created: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.get_error: Exception | None = None
        self.delete_error: Exception | None = None

    def create(self, machine_id, name, *, region=None):
        self.created.append((machine_id, name))
        self.items[name] = _Snapshot(name=name, status="ready")

    def get(self, name, *, org=None, region=None):
        if self.get_error is not None:
            raise self.get_error
        if name not in self.items:
            raise NotFoundError("snapshot not found")
        return self.items[name]

    def delete(self, name, *, org=None, region=None):
        if self.delete_error is not None:
            raise self.delete_error
        if name not in self.items:
            raise NotFoundError("snapshot not found")
        del self.items[name]
        self.deleted.append(name)


class FakeBoxd:
    instances: list[FakeBoxd] = []

    def __init__(self, *, api_key=None, token=None, region=None, **kw):
        self.api_key = api_key
        self.token = token
        self.region = region
        self.calls: list = []
        self.machines = _Machines(self)
        self.snapshots = _Snapshots(self)
        FakeBoxd.instances.append(self)


@pytest.fixture
def fake_boxd(monkeypatch):
    """Install a fake ``boxd`` package and a test key; yield the module."""
    errors = types.ModuleType("boxd.errors")
    for cls in (BoxdError, NotFoundError, ConflictError, AuthenticationError, PermissionDeniedError):
        setattr(errors, cls.__name__, cls)
    mod = types.ModuleType("boxd")
    mod.Boxd = FakeBoxd
    mod.errors = errors
    monkeypatch.setitem(sys.modules, "boxd", mod)
    monkeypatch.setitem(sys.modules, "boxd.errors", errors)
    for var in ("BOXD_API_KEY", "BOXD_TOKEN", "BOXD_ORG", "BOXD_REGION", "RLLM_BOXD_NAME_PREFIX", "RLLM_BOXD_LABELS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("BOXD_API_KEY", "test-key")
    FakeBoxd.instances = []
    return mod


def _client() -> FakeBoxd:
    return FakeBoxd.instances[-1]


def _sandbox(name="rllm-task-1-abc123", image="python:3.11-slim", **kw):
    from rllm.sandbox.backends.boxd import BoxdSandbox

    sb = BoxdSandbox(name=name, image=image, **kw)
    return sb, _client()


# ---------------------------------------------------------------------------
# Setup and configuration
# ---------------------------------------------------------------------------


def test_missing_boxd_sdk_raises_friendly_install_hint(monkeypatch):
    """Without the ``boxd`` package, construction names the install command."""
    monkeypatch.delitem(sys.modules, "boxd", raising=False)
    monkeypatch.delitem(sys.modules, "boxd.errors", raising=False)
    real_import = builtins.__import__

    def _block_boxd(name, *args, **kwargs):
        if name == "boxd" or name.startswith("boxd."):
            raise ModuleNotFoundError("No module named 'boxd'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_boxd)
    from rllm.sandbox.backends.boxd import BoxdSandbox

    with pytest.raises(ImportError, match="pip install boxd"):
        BoxdSandbox(name="test")


def test_missing_api_key_is_a_clear_error(fake_boxd, monkeypatch):
    monkeypatch.delenv("BOXD_API_KEY", raising=False)
    from rllm.sandbox.backends.boxd import BoxdSandbox

    with pytest.raises(ValueError, match="BOXD_API_KEY"):
        BoxdSandbox(name="test")
    assert FakeBoxd.instances == []  # no client, no API call


def test_api_key_kwarg_and_token_env(fake_boxd, monkeypatch):
    _sandbox(api_key="explicit")
    assert _client().api_key == "explicit"

    monkeypatch.delenv("BOXD_API_KEY")
    monkeypatch.setenv("BOXD_TOKEN", "tok")
    _sandbox()
    assert _client().token == "tok" and _client().api_key is None


def test_satisfies_sandbox_protocol(fake_boxd):
    sb, _ = _sandbox()
    assert isinstance(sb, Sandbox)


def test_create_from_image_is_isolated_by_default(fake_boxd):
    sb, c = _sandbox()
    (created,) = c.machines.created
    assert created["image"] == "python:3.11-slim"
    assert created["isolated"] is True
    assert created["auto_suspend_timeout"] == 0
    assert created["labels"] == {"app": "rllm"}
    assert "from_snapshot" not in created
    assert sb.machine_id == "m-1"
    assert ("wait", "m-1", 600.0) in c.calls


def test_create_passes_org_region_env_size_and_labels(fake_boxd, monkeypatch):
    monkeypatch.setenv("BOXD_ORG", "acme")
    monkeypatch.setenv("BOXD_REGION", "fra")
    monkeypatch.setenv("RLLM_BOXD_LABELS", "run=r1, team=rl")
    _, c = _sandbox(cpu=3, memory=6, env={"A": "1"}, labels={"run": "r2"}, auto_destroy_timeout=3600, create_timeout=30)
    (created,) = c.machines.created
    assert created["org"] == "acme"
    assert created["region"] == "fra"
    assert c.region == "fra"
    assert created["vcpu"] == 4  # 3 vCPU / 6 GiB rounds up to the 4x16 size
    assert "env" not in created
    assert created["labels"] == {"app": "rllm", "run": "r2", "team": "rl"}
    assert created["auto_destroy_timeout"] == 3600
    assert ("wait", "m-1", 30.0) in c.calls


def test_isolation_can_be_turned_off_explicitly(fake_boxd):
    _, c = _sandbox(isolated=False)
    assert c.machines.created[0]["isolated"] is False


@pytest.mark.parametrize(
    ("cpu", "mem", "expected"),
    [(None, None, None), (1, None, (1, 4)), (None, 4, (1, 4)), (2, 2, (2, 8)), (None, 8.5, (4, 16)), (8, 32, (8, 32)), (16, None, (8, 32))],
)
def test_size_rounds_up_to_a_boxd_size(cpu, mem, expected):
    from rllm.sandbox.backends.boxd import _size_for

    assert _size_for(cpu, mem) == expected


@pytest.mark.parametrize(
    ("name", "prefix", "expected"),
    [
        ("rllm-Task_1.2-abc123", "", "rllm-task-1-2-abc123"),
        ("rllm-env-0123456789ab-build", "t-rllm-", "t-rllm-rllm-env-0123456789ab-build"),
        ("123-task", "", "r123-task"),
    ],
)
def test_machine_name_is_sanitized(name, prefix, expected):
    from rllm.sandbox.backends.boxd import _machine_name

    assert _machine_name(name, prefix) == expected


def test_long_machine_name_keeps_the_unique_suffix():
    from rllm.sandbox.backends.boxd import _machine_name

    out = _machine_name("rllm-" + "very-long-task-id-" * 6 + "abc123")
    assert len(out) <= 63
    assert out.endswith("abc123")


def test_name_prefix_from_env(fake_boxd, monkeypatch):
    monkeypatch.setenv("RLLM_BOXD_NAME_PREFIX", "t-rllm-")
    sb, c = _sandbox(name="task-1")
    assert c.machines.created[0]["name"] == "t-rllm-task-1"
    assert sb.machine_name == "t-rllm-task-1"


def test_create_failure_while_waiting_deletes_the_machine(fake_boxd, monkeypatch):
    def boom(self, machine_id, **kw):
        raise TimeoutError("not ready")

    monkeypatch.setattr(_Machines, "wait_until_ready", boom)
    from rllm.sandbox.backends.boxd import BoxdSandbox

    with pytest.raises(TimeoutError):
        BoxdSandbox(name="t", image="python:3.11-slim")
    assert _client().machines.deleted == ["m-1"]


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def test_snapshot_name_restores_from_snapshot(fake_boxd):
    _, c = _sandbox(image="rllm-env-0123456789ab", cpu=4, memory=16, labels={"x": "y"})
    (created,) = c.machines.created
    assert created["from_snapshot"] == "rllm-env-0123456789ab"
    assert "image" not in created
    # A restore takes the snapshot's size and labels; only these may be passed.
    assert set(created) <= {"name", "from_snapshot", "org", "isolated", "auto_suspend_timeout", "region"}


def test_missing_snapshot_raises_snapshot_not_found(fake_boxd, monkeypatch):
    monkeypatch.setattr(_Machines, "create", lambda self, name=None, **kw: (_ for _ in ()).throw(NotFoundError("snapshot not found (region nyc)")))
    from rllm.sandbox.backends.boxd import BoxdSandbox

    with pytest.raises(SnapshotNotFound, match="rllm-env-gone"):
        BoxdSandbox(name="t", image="rllm-env-gone")


def test_missing_image_is_not_mistaken_for_a_snapshot(fake_boxd, monkeypatch):
    monkeypatch.setattr(_Machines, "create", lambda self, name=None, **kw: (_ for _ in ()).throw(NotFoundError("image not found")))
    from rllm.sandbox.backends.boxd import BoxdSandbox

    with pytest.raises(NotFoundError):
        BoxdSandbox(name="t", image="python:9.9-nope")


def test_ref_absent_only_on_confirmed_absence(fake_boxd):
    from rllm.sandbox.backends.boxd import _boxd_ref_absent

    FakeBoxd.instances = []
    orig_init = FakeBoxd.__init__
    state = {}

    def init(self, **kw):
        orig_init(self, **kw)
        self.snapshots.items = dict(state.get("items", {}))
        self.snapshots.get_error = state.get("get_error")

    FakeBoxd.__init__ = init
    try:
        state.update(items={"a": _Snapshot("a", "ready"), "f": _Snapshot("f", "failed")})
        assert _boxd_ref_absent("a") is False
        assert _boxd_ref_absent("f") is True
        assert _boxd_ref_absent("gone") is True
        state["get_error"] = AuthenticationError("bad key")
        assert _boxd_ref_absent("a") is False  # unsure: keep
    finally:
        FakeBoxd.__init__ = orig_init


def test_delete_snapshot_reports_confirmed_deletion_only(fake_boxd, monkeypatch):
    from rllm.sandbox.backends import boxd as backend

    client = FakeBoxd(api_key="k")
    client.snapshots.items = {"a": _Snapshot("a")}
    monkeypatch.setattr(backend, "_make_client", lambda **kw: client)
    assert backend.delete_boxd_snapshot("a") is True
    assert backend.delete_boxd_snapshot("a") is True  # already gone
    client.snapshots.delete_error = PermissionDeniedError("nope")
    assert backend.delete_boxd_snapshot("a") is False


def test_build_snapshot_reuses_ready_snapshot(fake_boxd, monkeypatch):
    from rllm.sandbox.backends import boxd as backend

    client = FakeBoxd(api_key="k")
    client.snapshots.items = {"rllm-env-k": _Snapshot("rllm-env-k", "ready")}
    monkeypatch.setattr(backend, "_make_client", lambda **kw: client)
    called = []
    monkeypatch.setattr("rllm.eval._resolution._create_base_sandbox", lambda *a, **k: called.append(1))
    assert backend.build_boxd_snapshot(object(), "rllm-env-k") == "rllm-env-k"
    assert called == []


def test_build_snapshot_captures_build_machine_and_closes_it(fake_boxd, monkeypatch):
    from rllm.sandbox.backends import boxd as backend

    client = FakeBoxd(api_key="k")
    monkeypatch.setattr(backend, "_make_client", lambda **kw: client)

    class _SB:
        machine_id = "m-build"
        closed = False
        execs: list = []

        def exec(self, command, timeout=None, user=None):
            self.execs.append((command, user))
            return ""

        def close(self):
            _SB.closed = True

    made = {}

    def fake_create(task, backend_name, *, image=None, name=None, **kw):
        made.update(backend=backend_name, name=name)
        return _SB()

    monkeypatch.setattr("rllm.eval._resolution._create_base_sandbox", fake_create)
    monkeypatch.setattr("rllm.eval._resolution._replay_dockerfile", lambda task, sb, b: sb.exec("RUN-STEP"))
    ref = backend.build_boxd_snapshot(object(), "rllm-env-new", install_script="install-cli", force=True)
    assert ref == "rllm-env-new"
    assert made == {"backend": "boxd", "name": "rllm-env-new-build"}
    assert _SB.execs == [("RUN-STEP", None), ("install-cli", "root")]
    assert client.snapshots.created == [("m-build", "rllm-env-new")]
    assert _SB.closed


def test_registry_dispatch_knows_boxd(fake_boxd, monkeypatch):
    from rllm.sandbox import sandboxed_flow
    from rllm.sandbox.backends import boxd as backend

    monkeypatch.setattr(backend, "delete_boxd_snapshot", lambda ref: ref == "x")
    monkeypatch.setattr(backend, "_boxd_ref_absent", lambda ref: ref == "y")
    assert sandboxed_flow.delete_snapshot("boxd", "x") is True
    assert sandboxed_flow.snapshot_absent("boxd", "y") is True
    sb = sandboxed_flow.create_sandbox("boxd", name="t", image="python:3.11-slim")
    assert type(sb).__name__ == "BoxdSandbox"
    with pytest.raises(ValueError, match="boxd"):
        sandboxed_flow.create_sandbox("nope", name="t", image="x")


def test_task_resources_map_to_boxd_kwargs():
    from pathlib import Path

    from rllm.eval._resolution import _sandbox_resource_kwargs
    from rllm.types import Task

    task = Task(id="t", instruction="", metadata={"environment": {"cpus": 2, "memory_mb": 6144, "storage_mb": 20480}}, dataset_dir=Path("."))
    assert _sandbox_resource_kwargs(task, "boxd") == {"cpu": 2, "memory": 6.0, "create_timeout": 600.0}


# ---------------------------------------------------------------------------
# exec
# ---------------------------------------------------------------------------


def test_exec_runs_bash_c_by_id_and_returns_stdout(fake_boxd):
    sb, c = _sandbox()
    c.machines.exec_results = [_Result(stdout="hi\n")]
    assert sb.exec("echo hi | cat", timeout=12) == "hi\n"
    machine_id, argv, timeout = c.machines.execs[-1]
    assert (machine_id, argv[:2], timeout) == ("m-1", ["bash", "-c"], 12)
    assert "bash -c 'echo hi | cat'" in argv[2]


def test_env_goes_with_every_exec(fake_boxd):
    sb, c = _sandbox(env={"A": 1, "B": "two"})
    sb.exec("echo $A")
    sb.exec("echo $B", user="root")
    assert c.machines.exec_env == [{"A": "1", "B": "two"}, {"A": "1", "B": "two"}]


def test_env_goes_with_exec_on_a_snapshot_restore(fake_boxd):
    sb, c = _sandbox(image="rllm-env-0123456789ab", env={"A": "1"})
    sb.exec("echo $A")
    assert c.machines.exec_env == [{"A": "1"}]


def test_no_env_sends_none(fake_boxd):
    sb, c = _sandbox()
    sb.exec("true")
    assert c.machines.exec_env == [None]


def test_exec_nonzero_exit_raises_with_stderr_tail(fake_boxd):
    sb, c = _sandbox()
    c.machines.exec_results = [_Result(stdout="partial", stderr="boom", exit_code=3)]
    with pytest.raises(RuntimeError, match=r"exit 3.*\n.*boom"):
        sb.exec("false")


def test_exec_timeout_raises_runtime_error(fake_boxd):
    sb, c = _sandbox()
    c.machines.exec_results = [BoxdError("exec timed out after 2s", code="deadline_exceeded")]
    with pytest.raises(RuntimeError, match="timed out after 2"):
        sb.exec("sleep 10", timeout=2)


def test_exec_other_api_errors_propagate(fake_boxd):
    sb, c = _sandbox()
    c.machines.exec_results = [AuthenticationError("bad key")]
    with pytest.raises(AuthenticationError):
        sb.exec("true")


def test_exec_on_stopped_machine_starts_it_and_retries(fake_boxd):
    sb, c = _sandbox()
    c.machines.exec_results = [ConflictError("machine is stopped"), _Result(stdout="ok")]
    assert sb.exec("echo ok") == "ok"
    assert c.machines.started == ["m-1"]
    assert len(c.machines.execs) == 2


def test_exec_as_user_switches_user_in_the_shell(fake_boxd):
    sb, c = _sandbox()
    sb.exec("whoami", user="root")
    (_, argv, _) = c.machines.execs[-1]
    assert argv[:2] == ["bash", "-c"]
    assert "runuser" in argv[2] and "sudo -n" in argv[2] and "'whoami'" in argv[2]


def test_safe_exec_returns_error_text(fake_boxd):
    from rllm.sandbox.protocol import _safe_exec

    sb, c = _sandbox()
    c.machines.exec_results = [_Result(stderr="nope", exit_code=1)]
    assert "nope" in _safe_exec(sb, "false")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _capture_exec(sb) -> list[str]:
    """Record the commands ``sb.exec`` receives (before user switch and keepalive wrapping)."""
    cmds: list[str] = []
    real = sb.exec

    def spy(command, timeout=None, user=None):
        cmds.append(command)
        return real(command, timeout=timeout, user=user)

    sb.exec = spy
    return cmds


def test_upload_file_uses_native_upload(fake_boxd, tmp_path):
    sb, c = _sandbox()
    f = tmp_path / "a.bin"
    f.write_bytes(b"\x00\x01data")
    sb.upload_file(str(f), "/app/sub/a.bin")
    assert c.machines.files.uploads["/app/sub/a.bin"] == b"\x00\x01data"


def test_upload_dir_uploads_one_tarball_and_extracts_it(fake_boxd, tmp_path):
    sb, c = _sandbox()
    cmds = _capture_exec(sb)
    (tmp_path / "tree" / "sub").mkdir(parents=True)
    (tmp_path / "tree" / "a.txt").write_text("a")
    (tmp_path / "tree" / "sub" / "b.sh").write_text("#!/bin/sh\n")
    (tmp_path / "tree" / "sub" / "b.sh").chmod(0o755)
    sb.upload_dir(str(tmp_path / "tree"), "/app/tests")

    (path,) = c.machines.files.uploads
    assert path.startswith("/tmp/_upload_") and path.endswith(".tar.gz")
    with tarfile.open(fileobj=io.BytesIO(c.machines.files.uploads[path]), mode="r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
    assert {"tests/a.txt", "tests/sub/b.sh"} <= set(members)
    assert members["tests/sub/b.sh"].mode & 0o111
    (cmd,) = cmds
    assert f"tar xzf {path} --no-same-owner -C '/app'" in cmd
    assert f"rm -f {path}" in cmd


def test_upload_dir_missing_local_path(fake_boxd, tmp_path):
    sb, _ = _sandbox()
    with pytest.raises(FileNotFoundError):
        sb.upload_dir(str(tmp_path / "nope"), "/app/x")


# ---------------------------------------------------------------------------
# is_alive / close
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "alive"),
    [("running", True), ("suspended", True), ("hibernated", True), ("starting", True), ("stopped", False), ("failed", False), ("destroyed", False)],
)
def test_is_alive_by_status(fake_boxd, status, alive):
    sb, c = _sandbox()
    c.machines.by_id["m-1"].status = status
    assert sb.is_alive() is alive


def test_is_alive_never_raises(fake_boxd, monkeypatch):
    sb, c = _sandbox()
    monkeypatch.setattr(_Machines, "get", lambda self, mid, **kw: (_ for _ in ()).throw(RuntimeError("network")))
    assert sb.is_alive() is False


def test_close_deletes_by_id_once(fake_boxd):
    sb, c = _sandbox()
    sb.close()
    sb.close()
    assert c.machines.deleted == ["m-1"]
    assert sb.is_alive() is False


def test_close_falls_back_to_hibernate_when_delete_fails(fake_boxd):
    sb, c = _sandbox()
    c.machines.delete_error = PermissionDeniedError("no delete")
    sb.close()
    assert c.machines.hibernated == ["m-1"]


def test_atexit_hook_deletes_unclosed_sandboxes(fake_boxd):
    from rllm.sandbox.backends.boxd import _terminate_all_live

    sb, c = _sandbox()
    _terminate_all_live()
    assert c.machines.deleted == ["m-1"]


def test_upload_dir_quotes_a_remote_path_with_spaces(fake_boxd, tmp_path):
    sb, c = _sandbox()
    cmds = _capture_exec(sb)
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "a.txt").write_text("a")
    sb.upload_dir(str(tmp_path / "t"), "/app/my dir/t")
    (cmd,) = cmds
    assert "mkdir -p '/app/my dir'" in cmd
    assert "-C '/app/my dir'" in cmd


# ---------------------------------------------------------------------------
# The generated shell, run by a real local bash
# ---------------------------------------------------------------------------


def _run_local(argv, **kw):
    import subprocess

    return subprocess.run(argv, capture_output=True, text=True, timeout=30, **kw)


def test_wrapped_command_keeps_stdout_stderr_and_exit_code():
    from rllm.sandbox.backends.boxd import _as_user

    r = _run_local(_as_user("echo out; echo 'it''s' >&2; exit 7", None))
    assert (r.stdout, r.stderr, r.returncode) == ("out\n", "its\n", 7)


def test_wrapped_command_as_the_current_user_is_a_no_op():
    import getpass
    import os

    from rllm.sandbox.backends.boxd import _as_user

    for who in (getpass.getuser(), str(os.getuid())):
        r = _run_local(_as_user("id -un; exit 3", who))
        assert (r.stdout.strip(), r.returncode) == (getpass.getuser(), 3)


def test_keepalive_writes_nul_bytes_to_stderr_while_silent(monkeypatch):
    from rllm.sandbox.backends import boxd as backend

    monkeypatch.setattr(backend, "_KEEPALIVE_INTERVAL_S", 1)
    r = _run_local(backend._as_user("sleep 3.5; echo done", None))
    assert r.stdout == "done\n" and r.returncode == 0
    assert r.stderr.count("\0") >= 2 and r.stderr.replace("\0", "") == ""


def test_keepalive_bytes_are_dropped_from_error_text(fake_boxd):
    sb, c = _sandbox()
    c.machines.exec_results = [_Result(stderr="\0\0real error\0", exit_code=1)]
    with pytest.raises(RuntimeError) as e:
        sb.exec("false")
    assert "\0" not in str(e.value) and "real error" in str(e.value)

"""U.7-U.8 — unit tests for rllm_sandbox_snapshot."""

from __future__ import annotations

import glob
import os
import shutil


def _make_fake_workdir(root, name: str) -> str:
    """Build a fake bwrap-style workdir with /app + /tmp children."""
    wd = root / f"rllm-bwrap-{name}"
    (wd / "app" / "x").mkdir(parents=True)
    (wd / "tmp").mkdir(parents=True)
    (wd / "app" / "x" / "note.txt").write_text("real content")
    (wd / "app" / "x" / "__pycache__").mkdir()
    (wd / "app" / "x" / "__pycache__" / "foo.pyc").write_text("bytecode")
    (wd / "tmp" / "log.txt").write_text("agent log")
    return str(wd)


def _fresh_sandbox_snapshot(monkeypatch, tmp_path):
    """Set up fake BwrapSandbox module + reload plugin."""
    import importlib
    import sys
    import types

    # Fake BwrapSandbox with a `close()` method that will be patched.
    class FakeBwrapSandbox:
        def __init__(self, name, workdir):
            self.name = name
            self._workdir = workdir
            self.closed = False

        def close(self):
            self.closed = True

    fake_module = types.ModuleType("rllm.sandbox.backends.bwrap")
    fake_module.BwrapSandbox = FakeBwrapSandbox  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rllm", types.ModuleType("rllm"))
    monkeypatch.setitem(sys.modules, "rllm.sandbox", types.ModuleType("rllm.sandbox"))
    monkeypatch.setitem(sys.modules, "rllm.sandbox.backends", types.ModuleType("rllm.sandbox.backends"))
    monkeypatch.setitem(sys.modules, "rllm.sandbox.backends.bwrap", fake_module)

    if shutil.which("rsync") is None:
        import pytest

        pytest.skip("rsync not installed")

    monkeypatch.setenv("RLLM_SANDBOX_SNAPSHOT_ROOT", str(tmp_path / "snapshots"))

    import rllm_sandbox_snapshot

    importlib.reload(rllm_sandbox_snapshot)
    return FakeBwrapSandbox, tmp_path / "snapshots"


def test_workspace_rsync_excludes_pycache(tmp_path, monkeypatch) -> None:
    """U.7: rsync copies real files but excludes __pycache__."""
    FakeBwrapSandbox, snap_root = _fresh_sandbox_snapshot(monkeypatch, tmp_path)

    wd = _make_fake_workdir(tmp_path, "task-42")
    sb = FakeBwrapSandbox("task-42", wd)
    sb.close()
    assert sb.closed

    subdirs = sorted(glob.glob(str(snap_root / "*")))
    assert subdirs, f"no snapshot created under {snap_root}"
    snap_dir = subdirs[0]

    note = os.path.join(snap_dir, "app", "x", "note.txt")
    assert os.path.exists(note), f"expected {note}"
    with open(note) as f:
        assert f.read() == "real content"

    # pycache excluded
    assert not os.path.exists(os.path.join(snap_dir, "app", "x", "__pycache__"))
    # /tmp copied
    log = os.path.join(snap_dir, "tmp", "log.txt")
    assert os.path.exists(log)


def test_uid_seq_counter(tmp_path, monkeypatch) -> None:
    """U.8: consecutive closes produce distinct persist dirs (seq counter)."""
    FakeBwrapSandbox, snap_root = _fresh_sandbox_snapshot(monkeypatch, tmp_path)

    wd1 = _make_fake_workdir(tmp_path, "same-name")
    sb1 = FakeBwrapSandbox("same-name", wd1)
    sb1.close()

    wd2 = _make_fake_workdir(tmp_path, "same-name-2")
    sb2 = FakeBwrapSandbox("same-name", wd2)  # deliberately reuse the name
    sb2.close()

    subdirs = sorted(glob.glob(str(snap_root / "*")))
    assert len(subdirs) == 2, f"expected 2 distinct snapshots, got {subdirs}"

    # Both start with monotonically-increasing seq prefix
    prefixes = [os.path.basename(d).split("_")[0] for d in subdirs]
    assert prefixes[0] < prefixes[1]


def test_snapshot_disabled_when_env_unset(tmp_path, monkeypatch) -> None:
    """RLLM_SANDBOX_SNAPSHOT_ROOT unset → snapshot is no-op, close still runs."""
    import importlib
    import sys
    import types

    class FakeBwrapSandbox:
        def __init__(self, name, workdir):
            self.name = name
            self._workdir = workdir
            self.closed = False

        def close(self):
            self.closed = True

    fake_module = types.ModuleType("rllm.sandbox.backends.bwrap")
    fake_module.BwrapSandbox = FakeBwrapSandbox  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rllm", types.ModuleType("rllm"))
    monkeypatch.setitem(sys.modules, "rllm.sandbox", types.ModuleType("rllm.sandbox"))
    monkeypatch.setitem(sys.modules, "rllm.sandbox.backends", types.ModuleType("rllm.sandbox.backends"))
    monkeypatch.setitem(sys.modules, "rllm.sandbox.backends.bwrap", fake_module)
    monkeypatch.delenv("RLLM_SANDBOX_SNAPSHOT_ROOT", raising=False)

    import rllm_sandbox_snapshot

    importlib.reload(rllm_sandbox_snapshot)

    wd = _make_fake_workdir(tmp_path, "noop")
    sb = FakeBwrapSandbox("noop", wd)
    sb.close()
    assert sb.closed  # close still runs (delegates to original)

"""Tests for the Modal sandbox backend wrapper.

Covers behavior that doesn't require the real ``modal`` SDK or a Modal
token: the uploads build shell commands by string interpolation, so the
remote paths they embed have to be quoted.
"""

from __future__ import annotations

import shlex

from rllm.sandbox.backends.modal_backend import ModalSandbox


def _sandbox_recording_execs(monkeypatch):
    """A ModalSandbox that records shell commands instead of running them.

    Bypasses ``__init__`` so no Modal App/Sandbox is created.
    """
    sb = object.__new__(ModalSandbox)
    sb.name = "test"
    commands: list[str] = []

    def _record(command: str) -> str:
        commands.append(command)
        return ""

    monkeypatch.setattr(sb, "_exec_unchecked", _record)
    return sb, commands


def test_upload_file_quotes_remote_path(monkeypatch, tmp_path):
    """A remote path with a space must land as one path, not two words.

    Unquoted, ``mkdir -p /w/my dir`` makes two directories and
    ``base64 -d > /w/my dir/f.txt`` fails with "Is a directory" — and
    the failure is swallowed by ``_exec_unchecked``, so the upload
    silently writes nothing.
    """
    local = tmp_path / "f.txt"
    local.write_text("hello")
    sb, commands = _sandbox_recording_execs(monkeypatch)

    sb.upload_file(str(local), "/workspace/my dir/f.txt")

    mkdir = next(c for c in commands if c.startswith("mkdir -p"))
    assert mkdir == f"mkdir -p {shlex.quote('/workspace/my dir')}"
    write = next(c for c in commands if "base64 -d >" in c)
    assert write.endswith(shlex.quote("/workspace/my dir/f.txt"))


def test_upload_dir_quotes_remote_parent(monkeypatch, tmp_path):
    """Same for the directory upload's mkdir and ``tar -C`` target."""
    src = tmp_path / "files"
    src.mkdir()
    (src / "a.txt").write_text("a")
    sb, commands = _sandbox_recording_execs(monkeypatch)

    sb.upload_dir(str(src), "/home/agent/my project/files")

    mkdir = next(c for c in commands if c.startswith("mkdir -p"))
    assert mkdir == f"mkdir -p {shlex.quote('/home/agent/my project')}"
    untar = next(c for c in commands if "tar xzf" in c)
    assert untar.endswith(f"-C {shlex.quote('/home/agent/my project')}")

from __future__ import annotations

import io
import tarfile

import pytest

from rllm.sandbox.backends.modal_backend import ModalSandbox


def _archive() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as bundle:
        content = b"presentation"
        item = tarfile.TarInfo("deliverables/nested/slides.pptx")
        item.size = len(content)
        bundle.addfile(item, io.BytesIO(content))

        traversal = tarfile.TarInfo("deliverables/../escape.txt")
        traversal.size = len(content)
        bundle.addfile(traversal, io.BytesIO(content))
    return stream.getvalue()


class _Reader:
    def __init__(self, value: bytes):
        self.value = value

    def read(self) -> bytes:
        return self.value


class _Process:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0):
        self.stdout = _Reader(stdout)
        self.stderr = _Reader(stderr)
        self.returncode = returncode
        self.waited = False

    def wait(self) -> None:
        self.waited = True


def test_download_dir_streams_binary_tar_and_extracts_safe_files(tmp_path):
    process = _Process(_archive())

    class Sandbox:
        def exec(self, *args, **kwargs):
            assert args == ("tar", "czf", "-", "-C", "/home/user", "deliverables")
            assert kwargs == {"text": False}
            return process

    sandbox = ModalSandbox.__new__(ModalSandbox)
    sandbox.name = "test-modal"
    sandbox._sandbox = Sandbox()

    downloaded = sandbox.download_dir("/home/user/deliverables/", str(tmp_path))

    expected = tmp_path / "nested" / "slides.pptx"
    assert downloaded == [str(expected)]
    assert expected.read_bytes() == b"presentation"
    assert not (tmp_path.parent / "escape.txt").exists()
    assert process.waited


def test_download_dir_surfaces_modal_tar_failure(tmp_path):
    process = _Process(b"", stderr=b"tar: deliverables: Cannot stat", returncode=2)

    class Sandbox:
        def exec(self, *args, **kwargs):
            return process

    sandbox = ModalSandbox.__new__(ModalSandbox)
    sandbox.name = "test-modal"
    sandbox._sandbox = Sandbox()

    with pytest.raises(RuntimeError, match="Cannot stat"):
        sandbox.download_dir("/home/user/deliverables", str(tmp_path))


def test_download_dir_rejects_filesystem_root(tmp_path):
    sandbox = ModalSandbox.__new__(ModalSandbox)

    with pytest.raises(ValueError, match="other than the filesystem root"):
        sandbox.download_dir("/", str(tmp_path))


def test_snapshot_capture_is_given_a_workable_timeout():
    """modal's snapshot_filesystem() defaults to 55s.

    That only ever fits a minimal image; a multi-GB one fails with a bare
    "Timeout expired" that names neither the phase nor the bound. The capture
    must be given an explicit budget, and the sandbox has to outlive it.
    """
    import inspect

    import modal

    from rllm.sandbox.backends import modal_backend

    sdk_default = inspect.signature(modal.Sandbox.snapshot_filesystem).parameters["timeout"].default
    assert sdk_default <= 60, "SDK default changed; revisit whether an override is still needed"

    assert modal_backend._SNAPSHOT_TIMEOUT >= 600
    source = inspect.getsource(modal_backend.build_modal_snapshot)
    assert "snapshot_filesystem(timeout=_SNAPSHOT_TIMEOUT)" in source
    # The sandbox is what is being captured, so its lifetime must cover it.
    assert "_SNAPSHOT_TIMEOUT + 600" in source

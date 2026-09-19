from __future__ import annotations

import io
import tarfile

from rllm.sandbox.backends.docker import DockerSandbox


def _archive() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as bundle:
        content = b"workbook"
        item = tarfile.TarInfo("deliverables/nested/Sample.xlsx")
        item.size = len(content)
        bundle.addfile(item, io.BytesIO(content))

        traversal = tarfile.TarInfo("deliverables/../escape.txt")
        traversal.size = len(content)
        bundle.addfile(traversal, io.BytesIO(content))
    return stream.getvalue()


def test_download_dir_extracts_regular_files_without_path_traversal(tmp_path):
    class Container:
        def get_archive(self, remote_path):
            assert remote_path == "/home/user/deliverables"
            return iter([_archive()]), {}

    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._container = Container()

    downloaded = sandbox.download_dir("/home/user/deliverables", str(tmp_path))

    expected = tmp_path / "nested" / "Sample.xlsx"
    assert downloaded == [str(expected)]
    assert expected.read_bytes() == b"workbook"
    assert not (tmp_path.parent / "escape.txt").exists()

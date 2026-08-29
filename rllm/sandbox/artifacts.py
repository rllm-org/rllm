"""Helpers for copying generated artifacts out of sandboxes."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO


def extract_regular_files(archive: BinaryIO, local_path: str, *, root_name: str) -> list[str]:
    """Safely extract regular files from a sandbox-created tar archive.

    Sandbox backends archive the requested directory itself, so ``root_name``
    is stripped before files are written under ``local_path``. Links, special
    files, empty paths, and paths that could escape the destination are ignored.
    """
    destination = Path(local_path)
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    downloaded: list[str] = []

    with tarfile.open(fileobj=archive, mode="r:*") as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if parts and parts[0] == root_name:
                parts = parts[1:]
            if not parts or any(part in {"", ".", ".."} for part in parts):
                continue
            target = destination.joinpath(*parts)
            if destination_resolved not in target.resolve().parents:
                continue
            source = bundle.extractfile(member)
            if source is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            downloaded.append(str(target))

    return downloaded

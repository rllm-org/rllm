"""Detect official Harbor CyberGym Level 1 task directories."""

from __future__ import annotations

from pathlib import Path

# Official Harbor generation does not copy ``description.txt`` onto the host.
# Docker ``ADD``s it from HuggingFace at image build. Detect the compose
# sidecar + Level 1 prompt instead of requiring host-side task_data/.
_REQUIRED_FILES = (
    "task.toml",
    "instruction.md",
    "environment/submit.sh",
    "environment/docker-compose.yaml",
    "environment/task-server/task_server.py",
    "tests/test.sh",
    "tests/verify.py",
)


def is_cybergym_task_dir(path: str | Path) -> bool:
    """True when *path* is a Harbor CyberGym Level 1 task directory.

    Requires the compose sidecar, ``submit.sh``, the post-episode verifier,
    and a Level 1 ``instruction.md`` (lists ``description.txt``; does not
    list Level 2 ``error.txt``). Does not require a ground-truth PoC or
    host-side ``environment/task_data/``.
    """
    root = Path(path)
    if not root.is_dir():
        return False
    if not all((root / rel).is_file() for rel in _REQUIRED_FILES):
        return False
    try:
        instruction = (root / "instruction.md").read_text(encoding="utf-8")
    except OSError:
        return False
    return "description.txt" in instruction and "bash /workspace/submit.sh" in instruction and "error.txt" not in instruction


def discover_cybergym_task_dirs(root: str | Path) -> list[Path]:
    """Return Level 1 task dirs at *root*, or in ``root/level1/``, or *root* itself."""
    path = Path(root)
    if not path.is_dir():
        return []
    if is_cybergym_task_dir(path):
        return [path.resolve()]

    found: list[Path] = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and is_cybergym_task_dir(child):
            found.append(child.resolve())
    if found:
        return found

    nested = path / "level1"
    if nested.is_dir() and nested != path:
        return discover_cybergym_task_dirs(nested)
    return []

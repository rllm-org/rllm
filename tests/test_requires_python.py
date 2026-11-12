"""The declared Python floor must cover the stdlib modules imported at module scope."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "rllm"


def _requires_python_floor() -> tuple[int, int]:
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        requires_python = tomllib.load(f)["project"]["requires-python"]
    match = re.search(r">=\s*(\d+)\.(\d+)", requires_python)
    assert match is not None, f"could not parse a floor from {requires_python!r}"
    return int(match.group(1)), int(match.group(2))


def test_requires_python_covers_tomllib():
    """``tomllib`` is 3.11+, so the declared floor cannot be 3.10.

    These are module-scope imports on the ``rllm eval`` path, so an install on
    an older interpreter fails at the first command rather than at install time.
    """
    users = [
        path.relative_to(_REPO_ROOT).as_posix()
        for path in sorted(_PACKAGE_ROOT.rglob("*.py"))
        if re.search(r"^\s*import tomllib", path.read_text(encoding="utf-8"), re.MULTILINE)
    ]
    assert users, "nothing imports tomllib any more — update or drop this test"
    assert _requires_python_floor() >= (3, 11), f"{users} import tomllib (3.11+) but requires-python allows older"

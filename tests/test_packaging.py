"""Packaging-metadata tests: what the wheel declares must match what the code needs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "rllm"


def _pyproject() -> dict:
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _declared_data_files() -> set[str]:
    """Package-relative paths of every file matched by [tool.setuptools.package-data]."""
    patterns = _pyproject()["tool"]["setuptools"]["package-data"]["rllm"]
    matched: set[str] = set()
    for pattern in patterns:
        for path in _PACKAGE_ROOT.glob(pattern):
            if path.is_file():
                matched.add(path.relative_to(_PACKAGE_ROOT).as_posix())
    return matched


def test_cli_templates_are_declared_as_package_data():
    """``rllm init`` opens rllm/cli/templates/*.tpl at runtime, so the wheel must ship them."""
    templates = sorted(p.relative_to(_PACKAGE_ROOT).as_posix() for p in (_PACKAGE_ROOT / "cli" / "templates").glob("*.tpl"))
    assert templates, "no .tpl templates found under rllm/cli/templates — update this test"
    missing = [t for t in templates if t not in _declared_data_files()]
    assert not missing, f"templates missing from [tool.setuptools.package-data]: {missing}"

"""Every advertised ``rllm[...]`` extra must actually be declared."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "rllm"


def _declared_extras() -> set[str]:
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        return set(tomllib.load(f)["project"]["optional-dependencies"])


def test_all_extra_references_only_declared_extras():
    """``pip install rllm[all]`` must not recurse into an extra that does not exist."""
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]
    assert "all" in extras, "the 'all' extra disappeared — update or drop this test"
    for requirement in extras["all"]:
        match = re.match(r"^rllm\[([^\]]+)\]", requirement.strip())
        if match is None:
            continue
        referenced = {name.strip() for name in match.group(1).split(",")}
        undeclared = sorted(referenced - set(extras))
        assert not undeclared, f"'all' references undeclared extras: {undeclared}"


def test_sources_advertise_only_declared_extras():
    """An error message that tells the user to install ``rllm[x]`` must name a real extra."""
    extras = _declared_extras()
    advertised: dict[str, set[str]] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        for group in re.findall(r"rllm\[([a-z0-9,\-]+)\]", path.read_text(encoding="utf-8")):
            undeclared = {name.strip() for name in group.split(",")} - extras
            if undeclared:
                advertised[path.relative_to(_REPO_ROOT).as_posix()] = undeclared
    assert not advertised, f"sources advertise undeclared extras: {advertised}"

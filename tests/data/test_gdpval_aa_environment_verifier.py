"""The build-time manifest check must fail loudly on any package drift.

``verify_environment.py`` is the only thing standing between a drifted Debian
snapshot (or a silently substituted wheel) and a GDPval run whose environment
no longer matches AA's published closure.
"""

from __future__ import annotations

import pytest

from rllm.data.gdpval_aa import verify_environment as ve


@pytest.fixture
def manifest(tmp_path):
    def write(name: str, body: str):
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        return str(path)

    return write


def test_reads_pinned_manifests(manifest):
    debian = ve._read_manifest(manifest("system.txt", "adduser=3.152\napt=3.0.3\n\n"), "=")
    python = ve._read_manifest(manifest("python.txt", "numpy==2.4.4\npandas==2.3.3\n"), "==")

    assert debian == {"adduser": "3.152", "apt": "3.0.3"}
    assert python == {"numpy": "2.4.4", "pandas": "2.3.3"}


def test_rejects_an_unpinned_entry(manifest):
    with pytest.raises(SystemExit, match="malformed pin"):
        ve._read_manifest(manifest("system.txt", "adduser\n"), "=")


def test_matching_closure_reports_no_problems():
    expected = {"numpy": "2.4.4", "pandas": "2.3.3"}

    assert ve._compare("python", expected, dict(expected), exempt=frozenset()) == []


def test_missing_package_is_a_problem():
    problems = ve._compare("debian", {"jq": "1.7.1-6"}, {}, exempt=frozenset())

    assert problems == ["debian: MISSING jq==1.7.1-6"]


def test_version_drift_is_a_problem():
    problems = ve._compare("debian", {"jq": "1.7.1-6"}, {"jq": "1.7.2-1"}, exempt=frozenset())

    assert problems == ["debian: VERSION MISMATCH jq: want 1.7.1-6, got 1.7.2-1"]


def test_unexpected_package_is_a_problem():
    """An extra package means the closure is not the published one."""
    problems = ve._compare("python", {}, {"requests": "2.32.0"}, exempt=frozenset())

    assert problems == ["python: UNEXPECTED requests==2.32.0"]


def test_virtualenv_bootstrap_packages_are_exempt():
    """pip and wheel seed the venv and are absent from AA's published freeze."""
    installed = {"numpy": "2.4.4", "pip": "25.3", "wheel": "0.45.1"}

    assert ve._compare("python", {"numpy": "2.4.4"}, installed, exempt=frozenset({"pip", "wheel"})) == []


def test_package_name_normalization_is_pep503():
    assert ve._normalize("Pillow") == "pillow"
    assert ve._normalize("python_docx") == "python-docx"
    assert ve._normalize("ruamel.yaml") == "ruamel-yaml"

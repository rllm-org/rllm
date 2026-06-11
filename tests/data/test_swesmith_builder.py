"""Tests for the rllm-swesmith benchmark builder (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import tomllib

from rllm.data.swesmith_builder import _write_dataset_toml, bug_in_test_file, patch_task_toml

_TASK_TOML = """\
version = "1.0"

[metadata]
instance_id = "demo__repo.abc123.func_basic__x1"

[verifier]
timeout_sec = 3000.0

[agent]
timeout_sec = 3000.0
"""


def _make_task(tmp_path: Path, *, patch_target: str, f2p_file: str) -> Path:
    task_dir = tmp_path / "demo__repo.abc123.func_basic__x1"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "task.toml").write_text(_TASK_TOML)
    cfg = {
        "patch": f"diff --git a/{patch_target} b/{patch_target}\n--- a/{patch_target}\n+++ b/{patch_target}\n",
        "FAIL_TO_PASS": [f"{f2p_file}::TestX::test_y"],
        "PASS_TO_PASS": [],
    }
    (task_dir / "tests" / "config.json").write_text(json.dumps(cfg))
    return task_dir


class TestBugInTestFile:
    def test_source_bug_is_solvable(self, tmp_path):
        task = _make_task(tmp_path, patch_target="src/repo/core.py", f2p_file="tests/test_core.py")
        assert not bug_in_test_file(task)

    def test_bug_in_f2p_test_file_is_unsolvable(self, tmp_path):
        # gpxpy-style: the whole suite lives in root test.py and the bug was
        # injected there — the F2P-removal commit deletes the file to fix.
        task = _make_task(tmp_path, patch_target="test.py", f2p_file="test.py")
        assert bug_in_test_file(task)


class TestPatchTaskToml:
    def test_adds_default_resources_when_absent(self, tmp_path):
        task = _make_task(tmp_path, patch_target="src/a.py", f2p_file="tests/test_a.py")
        patch_task_toml(task)
        meta = tomllib.loads((task / "task.toml").read_text())
        assert meta["environment"]["memory_mb"] == 4096
        # untouched upstream sections survive
        assert meta["verifier"]["timeout_sec"] == 3000.0

    def test_existing_environment_wins(self, tmp_path):
        task = _make_task(tmp_path, patch_target="src/a.py", f2p_file="tests/test_a.py")
        toml_path = task / "task.toml"
        toml_path.write_text(toml_path.read_text() + "\n[environment]\nmemory_mb = 8192\n")
        patch_task_toml(task)
        meta = tomllib.loads(toml_path.read_text())
        assert meta["environment"]["memory_mb"] == 8192


class TestDatasetToml:
    def test_loader_recognizes_output(self, tmp_path):
        _write_dataset_toml(tmp_path, name="rllm-swesmith", split="train", description="d", default_agent="mini-swe-agent")
        meta = tomllib.loads((tmp_path / "dataset.toml").read_text())
        assert meta["dataset"]["type"] == "sandbox"
        assert meta["dataset"]["default_agent"] == "mini-swe-agent"
        assert meta["verifier"]["script"] == "tests/test.sh"

"""Tests for the swebench_verified benchmark builder (no network, no swebench).

Covers the pure task-tree builders and the in-sandbox grading path
(grade.py + the vendored swebench log parser) — the parts that don't need
the ``swebench`` package or Docker. The ``build_benchmark`` HF/image path is
exercised separately during dataset pulls.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import tomllib

from rllm.data.swebench_verified_builder import (
    _ASSETS_DIR,
    _build_dockerfile,
    _build_solution_script,
    _build_task_toml,
    _decode_json_list,
    _write_dataset_toml,
)


class TestDecodeJsonList:
    def test_json_string(self):
        assert _decode_json_list('["a", "b"]') == ["a", "b"]

    def test_python_literal_string(self):
        assert _decode_json_list("['a', 'b']") == ["a", "b"]

    def test_passthrough_list(self):
        assert _decode_json_list(["x", "y"]) == ["x", "y"]

    def test_empty_and_none(self):
        assert _decode_json_list(None) == []
        assert _decode_json_list("") == []
        assert _decode_json_list("[]") == []


class TestBuildArtifacts:
    def test_dockerfile_clears_entrypoint(self):
        df = _build_dockerfile("swebench/sweb.eval.x86_64.django_1776_django-1:latest")
        assert df.startswith("FROM swebench/sweb.eval.x86_64.django_1776_django-1:latest")
        assert "ENTRYPOINT []" in df
        assert "WORKDIR /testbed" in df

    def test_task_toml_is_valid_and_targets_testbed(self):
        toml_str = _build_task_toml(
            instance_id="django__django-11133",
            repo="django/django",
            version="3.0",
            base_commit="deadbeef",
            image="swebench/sweb.eval.x86_64.django_1776_django-11133:latest",
        )
        meta = tomllib.loads(toml_str)
        assert meta["environment"]["workdir"] == "/testbed"
        assert meta["environment"]["docker_image"].startswith("swebench/sweb.eval.x86_64.django_1776_")
        assert meta["metadata"]["instance_id"] == "django__django-11133"
        assert meta["metadata"]["base_commit"] == "deadbeef"

    def test_solution_script_applies_gold_patch(self):
        script = _build_solution_script("deadbeef")
        assert 'git reset --hard "deadbeef"' in script
        assert "git apply -v /solution/gold.patch" in script
        assert "cd /testbed" in script

    def test_dataset_toml_is_sandbox_shape(self, tmp_path):
        _write_dataset_toml(tmp_path, name="swebench_verified", split="test", description="d", default_agent="mini-swe-agent")
        meta = tomllib.loads((tmp_path / "dataset.toml").read_text())
        assert meta["dataset"]["type"] == "sandbox"
        assert meta["dataset"]["default_agent"] == "mini-swe-agent"
        assert meta["verifier"]["script"] == "tests/test.sh"


def _run_grade(tmp_path: Path, *, repo: str, f2p: list[str], p2p: list[str], test_section: str) -> dict:
    """Run the shipped grade.py exactly as the in-sandbox verifier would."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    for fname in ("grade.py", "swebench_parsers.py"):
        (tests_dir / fname).write_text((_ASSETS_DIR / fname).read_text())
    inst = {"instance_id": "x", "repo": repo, "base_commit": "c", "FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p}
    (tests_dir / "instance.json").write_text(json.dumps(inst))
    log = f">>>>> Start Test Output\n{test_section}\n>>>>> End Test Output\n"
    (tmp_path / "eval.txt").write_text(log)
    reward_path = tmp_path / "reward.json"
    # grade.py prepends /tests to sys.path (absent here); PYTHONPATH lets it
    # still import the vendored parser from the tmp tests dir.
    subprocess.run(
        [sys.executable, str(tests_dir / "grade.py"), str(tmp_path / "eval.txt"), str(tests_dir / "instance.json"), str(reward_path)],
        check=True,
        env={"PYTHONPATH": str(tests_dir)},
    )
    return json.loads(reward_path.read_text())


class TestGrading:
    REPO = "pytest-dev/pytest"

    def test_all_pass_resolves(self, tmp_path):
        section = "PASSED t.py::test_a\nPASSED t.py::test_b\n"
        r = _run_grade(tmp_path, repo=self.REPO, f2p=["t.py::test_a"], p2p=["t.py::test_b"], test_section=section)
        assert r["reward"] == 1.0
        assert r["is_correct"] is True

    def test_f2p_failure_unresolved(self, tmp_path):
        section = "FAILED t.py::test_a\nPASSED t.py::test_b\n"
        r = _run_grade(tmp_path, repo=self.REPO, f2p=["t.py::test_a"], p2p=["t.py::test_b"], test_section=section)
        assert r["reward"] == 0.0
        assert r["metadata"]["f2p_missing"] == ["t.py::test_a"]

    def test_p2p_regression_unresolved(self, tmp_path):
        section = "PASSED t.py::test_a\nFAILED t.py::test_b\n"
        r = _run_grade(tmp_path, repo=self.REPO, f2p=["t.py::test_a"], p2p=["t.py::test_b"], test_section=section)
        assert r["reward"] == 0.0
        assert r["metadata"]["p2p_missing"] == ["t.py::test_b"]

    def test_xfail_counts_as_passed(self, tmp_path):
        # swebench's test_passed treats XFAIL as a pass.
        section = "XFAIL t.py::test_a\nPASSED t.py::test_b\n"
        r = _run_grade(tmp_path, repo=self.REPO, f2p=["t.py::test_a"], p2p=["t.py::test_b"], test_section=section)
        assert r["reward"] == 1.0

    def test_missing_test_is_failure(self, tmp_path):
        # F2P test absent from the log (e.g. collection error) → unresolved.
        section = "PASSED t.py::test_b\n"
        r = _run_grade(tmp_path, repo=self.REPO, f2p=["t.py::test_a"], p2p=["t.py::test_b"], test_section=section)
        assert r["reward"] == 0.0

"""Offline tests for the SWE-bench Pro sandbox builder."""

from __future__ import annotations

import json

from rllm.data import swebench_pro_builder as builder
from rllm.tasks.loader import BenchmarkLoader


def _row(instance_id: str = "instance_demo__repo-abc-v1") -> dict:
    return {
        "instance_id": instance_id,
        "repo": "demo/repo",
        "base_commit": "a" * 40,
        "patch": "diff --git a/source.py b/source.py\n--- a/source.py\n+++ b/source.py\n@@ -1 +1 @@\n-old\n+new\n",
        "test_patch": "",
        "problem_statement": "Fix the demo bug.",
        "requirements": "Keep the API stable.",
        "interface": "Function: solve()",
        "repo_language": "python",
        "fail_to_pass": '["tests/test_demo.py::test_fix"]',
        "pass_to_pass": "[]",
        "before_repo_set_cmd": f"git reset --hard {'a' * 40}\ngit checkout {'b' * 40} -- tests/test_demo.py",
        "selected_test_files_to_run": '["tests/test_demo.py"]',
        "dockerhub_tag": "demo.repo-demo__repo-abc-v1",
    }


def _scripts_tree(tmp_path, instance_id: str):
    root = tmp_path / "scripts-source"
    scripts = root / "run_scripts" / instance_id
    scripts.mkdir(parents=True)
    (scripts / "run_script.sh").write_text("#!/bin/bash\necho ok\n")
    (scripts / "parser.py").write_text("# synthetic parser\n")
    return root


def _mock_sources(monkeypatch, tmp_path, rows: list[dict]):
    scripts_root = _scripts_tree(tmp_path, rows[0]["instance_id"])
    monkeypatch.setattr(builder, "_resolve_hf_revision", lambda: "hf-revision")
    monkeypatch.setattr(builder, "_load_rows", lambda split, revision=None: (rows, "hf-fingerprint"))
    monkeypatch.setattr(builder, "_shallow_clone_scripts_repo", lambda: scripts_root)
    monkeypatch.setattr(builder, "_git_revision", lambda path: "scripts-head")
    monkeypatch.setattr(builder, "_git_submodule_revision", lambda path, name: "b" * 40)
    return scripts_root


def test_load_rows_uses_resolved_dataset_revision(monkeypatch):
    calls = []

    class FakeDataset(list):
        _fingerprint = "current-fingerprint"

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeDataset([_row()])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    assert builder._load_rows("test", revision="dataset-head") == ([_row()], "current-fingerprint")
    assert calls == [((builder.HF_REPO_ID,), {"split": "test", "revision": "dataset-head"})]


def test_resolve_hf_revision_uses_current_official_head(monkeypatch):
    class Info:
        sha = "dataset-head"

    class FakeApi:
        def dataset_info(self, repo_id):
            assert repo_id == builder.HF_REPO_ID
            return Info()

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    assert builder._resolve_hf_revision() == "dataset-head"


def test_scripts_clone_tracks_current_default_branch(monkeypatch, tmp_path):
    checkout = tmp_path / "scripts-checkout"
    checkout.mkdir()
    calls = []
    monkeypatch.setattr(builder.tempfile, "mkdtemp", lambda prefix: str(checkout))
    monkeypatch.setattr(builder.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))

    assert builder._shallow_clone_scripts_repo() == checkout
    assert calls == [
        (
            ["git", "clone", "--depth", "1", builder.SCRIPTS_REPO_URL, str(checkout)],
            {"check": True, "stdout": builder.subprocess.PIPE, "stderr": builder.subprocess.STDOUT},
        )
    ]


def test_strip_binary_hunks_matches_official_evaluator():
    patch = """\
diff --git a/source.py b/source.py
--- a/source.py
+++ b/source.py
@@ -1 +1 @@
-old
+new
diff --git a/logo.png b/logo.png
index 111..222 100644
GIT binary patch
literal 3
abc
diff --git a/readme.dat b/readme.dat
Binary files a/readme.dat and b/readme.dat differ
"""

    stripped = builder._strip_binary_hunks(patch)

    assert "source.py" in stripped
    assert "logo.png" not in stripped
    assert "readme.dat" not in stripped


def test_prune_stale_task_dirs_only_removes_task_directories(tmp_path):
    active = tmp_path / "active"
    stale = tmp_path / "stale"
    unrelated = tmp_path / "cache"
    for path in (active, stale):
        path.mkdir()
        (path / "task.toml").write_text("")
    unrelated.mkdir()

    assert builder._prune_stale_task_dirs(tmp_path, {"active"}) == 1
    assert active.exists()
    assert not stale.exists()
    assert unrelated.exists()


def test_build_materializes_current_release_and_provenance(monkeypatch, tmp_path):
    row = _row()
    _mock_sources(monkeypatch, tmp_path, [row])

    out = builder.build_benchmark(out_dir=tmp_path / "benchmark", register=False)

    loaded = BenchmarkLoader.load(str(out))
    assert loaded.harness_name == "swebench-pro-mini"
    assert len(loaded.tasks) == 1
    assert loaded.tasks[0].instruction == ("Fix the demo bug.\n\nRequirements:\nKeep the API stable.\n\nNew interfaces introduced:\nFunction: solve()")

    task = out / row["instance_id"]
    task_toml = (task / "task.toml").read_text()
    assert "ENTRYPOINT []" in (task / "environment" / "Dockerfile").read_text()
    assert "timeout_sec = 3600.0" in task_toml
    assert "b" * 40 in task_toml
    assert 'dataset_revision = "hf-revision"' in task_toml
    assert 'dataset_fingerprint = "hf-fingerprint"' in task_toml
    assert 'scripts_revision = "scripts-head"' in task_toml
    assert 'artifacts = ["/tmp/rllm/model_patch.diff"]' in task_toml
    assert 'environment_mode = "separate"' in task_toml
    assert "[[verifier.collect]]" in task_toml
    assert f"git diff --cached --binary {row['base_commit']}" in task_toml
    assert loaded.tasks[0].metadata["verifier_mode"] == "separate"
    assert loaded.tasks[0].metadata["artifacts"] == ["/tmp/rllm/model_patch.diff"]
    assert loaded.tasks[0].metadata["verifier_collect"][0]["timeout_sec"] == 300.0
    assert loaded.tasks[0].metadata["metadata"]["dataset_revision"] == "hf-revision"
    assert loaded.tasks[0].metadata["metadata"]["scripts_revision"] == "scripts-head"
    assert "Binary files" in (task / "tests" / "test.sh").read_text()
    assert "Using collected agent patch from separate verifier contract" in (task / "tests" / "test.sh").read_text()
    assert json.loads((task / "tests" / "instance.json").read_text())["fail_to_pass"] == ["tests/test_demo.py::test_fix"]

    manifest = json.loads((out / "source_manifest.json").read_text())
    assert manifest == {
        "dataset_source": builder.HF_REPO_ID,
        "dataset_revision": "hf-revision",
        "dataset_fingerprint": "hf-fingerprint",
        "dataset_split": "test",
        "upstream_task_count": 1,
        "scripts_source": builder.SCRIPTS_REPO_URL,
        "scripts_revision": "scripts-head",
        "official_mini_swe_agent_revision": "b" * 40,
        "materialized_task_count": 1,
        "materialized_task_ids_sha256": builder._task_ids_sha256([row["instance_id"]]),
        "materialized_task_ids": [row["instance_id"]],
        "skipped_task_ids": [],
    }


def test_full_build_prunes_stale_tasks(monkeypatch, tmp_path):
    row = _row()
    out = tmp_path / "benchmark"
    stale = out / "removed-upstream"
    stale.mkdir(parents=True)
    (stale / "task.toml").write_text("")
    _mock_sources(monkeypatch, tmp_path, [row])

    builder.build_benchmark(out_dir=out, register=False)

    assert not stale.exists()


def test_partial_build_preserves_unselected_tasks(monkeypatch, tmp_path):
    row = _row()
    out = tmp_path / "benchmark"
    unselected = out / "unselected"
    unselected.mkdir(parents=True)
    (unselected / "task.toml").write_text("")
    _mock_sources(monkeypatch, tmp_path, [row])

    builder.build_benchmark(out_dir=out, limit=1, register=False)

    assert unselected.exists()

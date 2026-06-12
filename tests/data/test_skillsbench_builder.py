"""Tests for the SkillsBench builder.

These don't hit the HuggingFace network — they patch ``hf_hub_download``
and feed a synthetic parquet straight to the builder so we can verify
the on-disk layout matches what ``BenchmarkLoader`` expects.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rllm.data import skillsbench_builder as sb


def _make_synthetic_parquet(parquet_path: Path) -> None:
    """Write a 3-row parquet that exercises every branch of the builder.

    Row 0 (``hello-world``): minimal text-only task, no files[], no skills.
    Row 1 (``with-fixtures``): includes an extra text file under ``data/`` and
        a binary blob under ``fixtures/`` to exercise the ``files`` expansion
        + the text-vs-binary split.
    Row 2 (``skill-task``): carries two skills entries to exercise the
        ``skills/<name>.md`` writeout.
    """
    rows = [
        {
            "task_id": "hello-world",
            "category": "demo",
            "difficulty": "easy",
            "difficulty_explanation": "",
            "tags": ["demo"],
            "allow_internet": False,
            "instruction": "Print hello-world.\n",
            "task_toml": ('schema_version = "1.2"\n[task]\nname = "skillsbench/hello-world"\ndescription = "demo"\n[environment]\nallow_internet = false\n'),
            "dockerfile": "FROM python:3.11-slim\nWORKDIR /workspace\n",
            "solve_sh": "#!/bin/bash\necho ok > /workspace/out.txt\n",
            "test_sh": "#!/bin/bash\ntest -f /workspace/out.txt\n",
            "test_outputs": "# placeholder fixtures\n",
            "skills": [],
            "files": [],
        },
        {
            "task_id": "with-fixtures",
            "category": "demo",
            "difficulty": "medium",
            "difficulty_explanation": "",
            "tags": [],
            "allow_internet": False,
            "instruction": "Use the fixtures.\n",
            "task_toml": ('schema_version = "1.2"\n[task]\nname = "skillsbench/with-fixtures"\ndescription = "demo"\n'),
            "dockerfile": "FROM python:3.11-slim\n",
            "solve_sh": "#!/bin/bash\nexit 0\n",
            "test_sh": "#!/bin/bash\nexit 0\n",
            "test_outputs": "",
            "skills": [],
            "files": [
                {
                    "path": "data/notes.txt",
                    "content": "hello",
                    "is_text": True,
                    "sha256": "",
                    "size_bytes": 5,
                },
                {
                    "path": "fixtures/blob.bin",
                    "content": b"\x00\x01\x02\x03",
                    "is_text": False,
                    "sha256": "",
                    "size_bytes": 4,
                },
            ],
        },
        {
            "task_id": "skill-task",
            "category": "demo",
            "difficulty": "hard",
            "difficulty_explanation": "",
            "tags": [],
            "allow_internet": False,
            "instruction": "Solve with skills.\n",
            "task_toml": ('schema_version = "1.2"\n[task]\nname = "skillsbench/skill-task"\ndescription = "demo"\n'),
            "dockerfile": "FROM python:3.11-slim\n",
            "solve_sh": "#!/bin/bash\nexit 0\n",
            "test_sh": "#!/bin/bash\nexit 0\n",
            "test_outputs": "",
            "skills": [
                {"name": "alpha", "description": "alpha skill", "skill_md": "# alpha\nbody"},
                {"name": "beta/slash", "description": "beta skill", "skill_md": "# beta\nbody"},
            ],
            "files": [],
        },
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path)


@pytest.fixture
def synthetic_parquet(tmp_path):
    """Write a synthetic parquet and patch hf_hub_download to return it."""
    parquet_path = tmp_path / "skillsbench-tasks.parquet"
    _make_synthetic_parquet(parquet_path)
    with patch.object(sb, "hf_hub_download", return_value=str(parquet_path), create=True):
        # The builder imports hf_hub_download inside the function body, so
        # patch the import path instead.
        with patch("huggingface_hub.hf_hub_download", return_value=str(parquet_path)):
            yield parquet_path


def test_builder_writes_canonical_layout(tmp_path, synthetic_parquet):
    out = tmp_path / "skillsbench_out"
    sb.build_benchmark(
        name="skillsbench",
        split="train",
        out_dir=out,
        register=False,
    )

    # dataset.toml at the root, recognized by BenchmarkLoader
    dataset_toml = out / "dataset.toml"
    assert dataset_toml.exists()
    text = dataset_toml.read_text()
    assert 'name = "skillsbench"' in text
    assert 'type = "sandbox"' in text

    # Each task ships the canonical Harbor-format files
    hello = out / "hello-world"
    assert (hello / "task.toml").exists()
    assert (hello / "instruction.md").read_text() == "Print hello-world.\n"
    assert (hello / "environment" / "Dockerfile").exists()
    assert (hello / "tests" / "test.sh").exists()
    assert (hello / "tests" / "test_outputs.py").exists()
    assert (hello / "solution" / "solve.sh").exists()

    # Executable bits land on the right scripts
    assert os.access(hello / "solution" / "solve.sh", os.X_OK)
    assert os.access(hello / "tests" / "test.sh", os.X_OK)


def test_builder_expands_files_list(tmp_path, synthetic_parquet):
    out = tmp_path / "skillsbench_out"
    sb.build_benchmark(name="skillsbench", split="train", out_dir=out, register=False)

    task = out / "with-fixtures"
    notes = task / "data" / "notes.txt"
    blob = task / "fixtures" / "blob.bin"

    assert notes.read_text() == "hello"
    assert blob.read_bytes() == b"\x00\x01\x02\x03"


def test_builder_writes_skills_into_build_context(tmp_path, synthetic_parquet):
    """Skills land inside environment/ so the Dockerfile COPY picks them up."""
    out = tmp_path / "skillsbench_out"
    sb.build_benchmark(name="skillsbench", split="train", out_dir=out, register=False)

    skills_root = out / "skill-task" / "environment" / "skills"
    assert (skills_root / "alpha" / "SKILL.md").read_text().startswith("# alpha")
    # The "beta/slash" name has a path separator → sanitized in the directory name.
    skill_dirs = sorted(p.name for p in skills_root.iterdir() if p.is_dir())
    assert "alpha" in skill_dirs
    # The sanitized name for "beta/slash" should also exist, but not as
    # nested directories.
    assert not (skills_root / "beta").exists()
    other = [d for d in skill_dirs if d != "alpha"]
    assert len(other) == 1
    assert (skills_root / other[0] / "SKILL.md").read_text().startswith("# beta")


def test_builder_patches_dockerfile_with_neutral_path_and_symlinks(tmp_path, synthetic_parquet):
    """The Dockerfile gains a neutral COPY + symlinks for every agent path."""
    out = tmp_path / "skillsbench_out"
    sb.build_benchmark(name="skillsbench", split="train", out_dir=out, register=False)

    dockerfile = (out / "skill-task" / "environment" / "Dockerfile").read_text()
    # Neutral copy — single source of truth in the image.
    assert "COPY skills /opt/skills/" in dockerfile
    # Symlinks land at every well-known agent discovery path.
    assert "ln -sfn /opt/skills /root/.claude/skills" in dockerfile
    assert "ln -sfn /opt/skills /root/.agents/skills" in dockerfile
    assert "ln -sfn /opt/skills /root/.terminus/skills" in dockerfile
    assert "ln -sfn /opt/skills /etc/codex/skills" in dockerfile
    assert sb._SKILLS_COPY_MARKER in dockerfile

    # Tasks with no skills get no patch — no spurious COPY/RUN lines.
    hello_dockerfile = (out / "hello-world" / "environment" / "Dockerfile").read_text()
    assert "COPY skills" not in hello_dockerfile
    assert "/opt/skills" not in hello_dockerfile


def test_builder_omits_skills_when_disabled(tmp_path, synthetic_parquet):
    out = tmp_path / "skillsbench_out"
    sb.build_benchmark(
        name="skillsbench-no-skills",
        split="train",
        out_dir=out,
        include_skills=False,
        register=False,
    )

    skills_root = out / "skill-task" / "environment" / "skills"
    assert not skills_root.exists()
    # Dockerfile must not reference /opt/skills — otherwise Docker would fail
    # to build because the source directory is missing.
    dockerfile = (out / "skill-task" / "environment" / "Dockerfile").read_text()
    assert "/opt/skills" not in dockerfile
    assert "COPY skills" not in dockerfile


def test_builder_respects_task_ids_filter(tmp_path, synthetic_parquet):
    out = tmp_path / "skillsbench_out"
    sb.build_benchmark(
        name="skillsbench",
        split="train",
        out_dir=out,
        task_ids=["hello-world"],
        register=False,
    )

    assert (out / "hello-world").is_dir()
    assert not (out / "with-fixtures").exists()
    assert not (out / "skill-task").exists()


def test_builder_respects_limit(tmp_path, synthetic_parquet):
    out = tmp_path / "skillsbench_out"
    sb.build_benchmark(
        name="skillsbench",
        split="train",
        out_dir=out,
        limit=2,
        register=False,
    )

    task_dirs = sorted(p.name for p in out.iterdir() if p.is_dir())
    # Parquet preserves row order, so we keep the first two.
    assert task_dirs == ["hello-world", "with-fixtures"]


def test_builder_output_is_loadable_by_benchmark_loader(tmp_path, synthetic_parquet):
    """End-to-end: the directory the builder produces must round-trip through
    rLLM's BenchmarkLoader as a sandbox dataset. This is the real contract."""
    from rllm.tasks.loader import BenchmarkLoader

    out = tmp_path / "skillsbench_out"
    sb.build_benchmark(name="skillsbench", split="train", out_dir=out, register=False)

    assert BenchmarkLoader.is_local_benchmark(str(out))
    result = BenchmarkLoader.load(str(out))

    assert result.name == "skillsbench"
    assert result.split == "train"
    task_ids = sorted(t.id for t in result.tasks)
    assert task_ids == sorted(["skillsbench/hello-world", "skillsbench/with-fixtures", "skillsbench/skill-task"])
    # Per-task instruction text rides through from instruction.md.
    by_id = {t.id: t for t in result.tasks}
    assert by_id["skillsbench/hello-world"].instruction.strip() == "Print hello-world."


def test_clean_rebuilds_from_scratch(tmp_path, synthetic_parquet):
    out = tmp_path / "skillsbench_out"
    sb.build_benchmark(name="skillsbench", split="train", out_dir=out, register=False)

    # Drop a sentinel file the second build should remove
    sentinel = out / "stale-task" / "task.toml"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("stale")

    sb.build_benchmark(name="skillsbench", split="train", out_dir=out, clean=True, register=False)

    assert not sentinel.exists()
    assert (out / "hello-world").exists()

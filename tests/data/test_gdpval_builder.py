from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import patch

from rllm.data import gdpval_aa as aa
from rllm.data import gdpval_builder as gb
from rllm.tasks.loader import BenchmarkLoader


def _rows() -> list[dict]:
    return [
        {
            "task_id": "task-one",
            "sector": "Finance",
            "occupation": "Analyst",
            "prompt": "Review the workbook and produce a report.",
            "reference_files": ["reference_files/source.xlsx"],
            "deliverable_files": ["deliverable_files/expert.docx"],
            "rubric_pretty": "Prefer a correct and polished report.",
            "rubric_json": "[]",
        },
        {
            "task_id": "task-without-gold",
            "sector": "Technology",
            "occupation": "Engineer",
            "prompt": "Create a design.",
            "reference_files": [],
            "deliverable_files": [],
            "rubric_pretty": "",
            "rubric_json": "[]",
        },
    ]


def _build(tmp_path: Path, *, limit: int | None = None) -> Path:
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"source workbook")
    expert = tmp_path / "expert.docx"
    expert.write_bytes(b"expert report")
    files = {
        "reference_files/source.xlsx": str(source),
        "deliverable_files/expert.docx": str(expert),
    }

    with (
        patch("datasets.load_dataset", return_value=_rows()),
        patch("huggingface_hub.hf_hub_download", side_effect=lambda _repo, path, **_kwargs: files[path]),
        patch.object(gb, "_dataset_revision", return_value="abc123"),
    ):
        out = tmp_path / "gdpval"
        gb.build_benchmark(out_dir=out, limit=limit, repair_office_files=False, register=False)
    return out


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_builder_writes_rllm_harbor_layout(tmp_path):
    out = _build(tmp_path)
    task = out / "task-one"

    assert (out / "dataset.toml").exists()
    for name in ["task.toml", "instruction.md", "gdpval_aa.json"]:
        assert (task / name).exists(), name
    assert (task / "environment" / "Dockerfile").exists()
    # No in-sandbox verifier: grading is the dataset-level reward_fn.
    assert not (task / "tests" / "test.sh").exists()
    assert not (task / "tests" / "evaluate.py").exists()


def test_dataset_declares_the_rubric_reward_fn(tmp_path):
    """``dataset.toml`` names the same grader the catalog does.

    The eval CLI resolves a benchmark either from the catalog or from this file
    once the directory exists on disk. If the two disagree — or this one is
    absent — the run silently grades with whatever else it can find.
    """
    dataset_toml = (_build(tmp_path) / "dataset.toml").read_text()

    assert '[verifier]\nname = "gdpval_rubric_reward_fn"' in dataset_toml


def test_dataset_verifier_matches_the_catalog_entry():
    """The generated ``[verifier]`` and ``datasets.json`` cannot drift apart."""
    catalog = json.loads((Path(gb.__file__).parent.parent / "registry" / "datasets.json").read_text())

    assert catalog["datasets"]["gdpval"]["reward_fn"] == gb.DEFAULT_REWARD_FN


def test_tasks_declare_no_verifier_of_their_own(tmp_path):
    """A per-task verifier would be a second grader that disagrees with the first."""
    task_toml = (_build(tmp_path, limit=1) / "task-one" / "task.toml").read_text()

    assert "[verifier]" not in task_toml


def test_expert_deliverables_never_enter_the_solver_environment(tmp_path):
    out = _build(tmp_path)
    task = out / "task-one"

    # Inputs are staged into environment/files (uploaded to the workdir);
    # expert deliverables stay under tests/ on the host. Nothing in the sandbox
    # reads them, and no verifier uploads tests/, so they never leave the host.
    assert (task / "environment" / "files" / "source.xlsx").read_bytes() == b"source workbook"
    assert (task / "tests" / "reference" / "expert.docx").read_bytes() == b"expert report"

    staged = [path.name for path in (task / "environment").rglob("*") if path.is_file()]
    assert "expert.docx" not in staged
    assert not any("expert" in name for name in staged)


def test_solver_facing_files_never_name_the_expert_deliverable(tmp_path):
    out = _build(tmp_path)
    task = out / "task-one"

    for name in ["instruction.md", "task.toml"]:
        text = (task / name).read_text()
        assert "expert.docx" not in text, name
        assert "deliverable_files" not in text, name


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_instruction_is_aa_prompt_with_absolute_reference_paths(tmp_path):
    out = _build(tmp_path)

    instruction = (out / "task-one" / "instruction.md").read_text()

    assert instruction == aa.render_aa_gdpval_task_prompt(
        "Review the workbook and produce a report.",
        ["/home/user/source.xlsx"],
    )
    assert "- /home/user/source.xlsx" in instruction
    # The old harness prompt listed bare basenames and asked for relative paths.
    assert "- source.xlsx" not in instruction
    assert "relative paths" not in instruction
    assert "Expected deliverable filename" not in instruction


def test_instruction_for_a_task_without_reference_files(tmp_path):
    out = _build(tmp_path)

    instruction = (out / "task-without-gold" / "instruction.md").read_text()

    assert instruction == aa.render_aa_gdpval_task_prompt("Create a design.", [])
    assert "## Reference Files Location" not in instruction


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------


def test_task_config_runs_the_solver_as_the_non_root_aa_user(tmp_path):
    out = _build(tmp_path, limit=1)

    task_toml = (out / "task-one" / "task.toml").read_text()

    assert '[agent]\nuser = "user"' in task_toml
    assert 'user = "root"' not in task_toml
    assert 'workdir = "/home/user"' in task_toml
    assert 'reference_files = ["/home/user/source.xlsx"]' in task_toml


def test_generated_tasks_carry_no_judge_credentials(tmp_path):
    out = _build(tmp_path, limit=1)

    task_toml = (out / "task-one" / "task.toml").read_text()

    assert "[verifier.env]" not in task_toml
    for name in ["GDPVAL_JUDGE_API_KEY", "GDPVAL_JUDGE_MODEL", "GDPVAL_JUDGE_BASE_URL"]:
        assert name not in task_toml, name


def test_provenance_records_hashes_and_pinned_environment(tmp_path):
    out = _build(tmp_path)

    provenance = json.loads((out / "task-one" / "gdpval_aa.json").read_text())

    assert provenance["dataset_repo"] == "openai/gdpval"
    assert provenance["dataset_revision"] == "abc123"
    assert provenance["sandbox_image_digest"] == aa.AA_BASE_IMAGE_DIGEST
    assert provenance["sandbox_platform"] == "linux/amd64"
    assert provenance["system_prompt_sha256"] == aa.sha256_text(aa.AA_GDPVAL_SYSTEM_PROMPT)
    assert provenance["task_prompt_sha256"] == aa.sha256_text((out / "task-one" / "instruction.md").read_text())
    assert [entry["path"] for entry in provenance["reference_files"]] == ["/home/user/source.xlsx"]


def test_provenance_reference_hash_matches_the_staged_file(tmp_path):
    out = _build(tmp_path)
    provenance = json.loads((out / "task-one" / "gdpval_aa.json").read_text())
    staged = out / "task-one" / "environment" / "files" / "source.xlsx"

    entry = provenance["reference_files"][0]

    assert entry["sha256"] == gb._sha256(staged)
    assert entry["size_bytes"] == staged.stat().st_size


def test_builder_output_round_trips_through_benchmark_loader(tmp_path):
    out = _build(tmp_path)

    result = BenchmarkLoader.load(str(out))

    assert result.name == "gdpval"
    assert result.harness_name == "stirrup"
    assert sorted(task.id for task in result.tasks) == ["task-one", "task-without-gold"]
    by_id = {task.id: task for task in result.tasks}
    assert by_id["task-one"].metadata["reference_files"] == ["/home/user/source.xlsx"]
    assert by_id["task-one"].metadata["agent_user"] == "user"
    assert by_id["task-one"].metadata["workdir"] == "/home/user"
    assert by_id["task-one"].metadata["environment"]["docker_image"] == aa.published_image_ref()


def test_builder_respects_limit(tmp_path):
    out = _build(tmp_path, limit=1)

    task_dirs = sorted(path.name for path in out.iterdir() if path.is_dir() and not path.name.startswith("."))
    assert task_dirs == ["task-one"]


def test_dataset_catalog_entry_points_to_builder():
    catalog_path = Path(gb.__file__).parents[1] / "registry" / "datasets.json"
    catalog = json.loads(catalog_path.read_text())

    entry = catalog["datasets"]["gdpval"]
    assert entry["builder"] == "rllm.data.gdpval_builder:build_benchmark"
    assert entry["default_agent"] == "gdpval-stirrup"


# ---------------------------------------------------------------------------
# Office repair
# ---------------------------------------------------------------------------


def _write_repairable_docx(path: Path) -> None:
    relationships = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="officeDocument" Target="word/document.xml"/>
</Relationships>"""
    document_relationships = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
</Relationships>"""
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        package.writestr("_rels/.rels", relationships)
        package.writestr("word/document.xml", "<document/>")
        package.writestr("word/settings.xml", '<w:settings xmlns:w="w" xmlns:ns1="broken"><ns1:item/></w:settings>')
        package.writestr("word/_rels/document.xml.rels", document_relationships)


def test_office_repair_is_auditable_and_preserves_original(tmp_path):
    source = tmp_path / "broken.docx"
    backup = tmp_path / "backups" / "broken.docx"
    _write_repairable_docx(source)
    original = source.read_bytes()

    record = gb._repair_office_file(source, backup)

    assert record["status"] == "repaired"
    assert backup.read_bytes() == original
    assert record["original_sha256"] != record["repaired_sha256"]
    assert record["original_sha256"] and record["repaired_sha256"]
    with zipfile.ZipFile(source) as package:
        assert "word/settings.xml" not in package.namelist()


def test_office_repair_leaves_intact_files_untouched(tmp_path):
    intact = tmp_path / "fine.xlsx"
    with zipfile.ZipFile(intact, "w") as package:
        package.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        package.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>')
    before = intact.read_bytes()

    record = gb._repair_office_file(intact, tmp_path / "backups" / "fine.xlsx")

    assert intact.read_bytes() == before
    assert record["original_sha256"] == record["repaired_sha256"]
    assert record["backup"] is None


def test_pull_mode_boots_the_published_image_without_replay(tmp_path):
    """With an image published, the image *is* the environment.

    ``replay_dockerfile`` must be false so every backend boots it as-is, and the
    task Dockerfile is a thin wrapper with no build steps — matching how the
    other rLLM benchmarks work.
    """
    out = _build(tmp_path, limit=1)
    task_toml = (out / "task-one" / "task.toml").read_text()
    dockerfile = (out / "task-one" / "environment" / "Dockerfile").read_text()

    assert "replay_dockerfile = false" in task_toml
    assert aa.published_image_ref() in task_toml
    # A digest pin, not just a moving tag: a re-push must not silently change
    # the environment of an already-materialized dataset.
    assert "@sha256:" in task_toml
    assert "RUN " not in dockerfile
    assert dockerfile.count("FROM ") == 1
    # The published manifest is an amd64-only index. Without the platform flag,
    # Docker on an arm64 host resolves the index against its own arch and dies
    # with "no match for platform in manifest" rather than emulating.
    assert f"FROM --platform={aa.AA_PLATFORM} " in dockerfile


def test_build_mode_keeps_replay_on_when_nothing_is_published(tmp_path, monkeypatch):
    """Without a published image the declared image is only the base.

    Replay must stay on: declaring an image otherwise defaults it off (Harbor's
    convention), which boots bare Debian with no solver user or packages while
    the snapshot still reports success.
    """
    monkeypatch.setattr(aa, "AA_PUBLISHED_IMAGE_DIGEST", "")
    monkeypatch.delenv("RLLM_GDPVAL_IMAGE", raising=False)
    out = _build(tmp_path, limit=1)

    task_toml = (out / "task-one" / "task.toml").read_text()
    dockerfile = (out / "task-one" / "environment" / "Dockerfile").read_text()
    assert "replay_dockerfile = true" in task_toml
    assert aa.AA_BASE_IMAGE_DIGEST in task_toml
    assert dockerfile.count("RUN ") == 14

    # Both modes emit `FROM --platform=...`; the loader must not read that flag
    # as the image reference.
    from rllm.tasks.loader import BenchmarkLoader

    assert "FROM --platform=" in dockerfile
    task = next(t for t in BenchmarkLoader.load(str(out)).tasks if t.id == "task-one")
    assert task.metadata["environment"]["docker_image"].startswith("debian:")


def test_published_image_can_be_overridden_for_a_mirror(monkeypatch):
    monkeypatch.setenv("RLLM_GDPVAL_IMAGE", "registry.internal/gdpval@sha256:abc")
    assert aa.published_image_ref() == "registry.internal/gdpval@sha256:abc"


# --------------------------------------------------------------------------- #
# Rubric plumbing for the ``gdpval`` grader
# --------------------------------------------------------------------------- #

_REAL_RUBRIC = json.dumps(
    [
        {"score": 5, "criterion": "The recommended sample size is 220.", "rubric_item_id": "x"},
        {"score": -2, "criterion": "The deliverable fabricates figures.", "rubric_item_id": "y"},
    ]
)


def _build_with_rubric(tmp_path: Path) -> Path:
    rows = _rows()
    rows[0]["rubric_json"] = _REAL_RUBRIC
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"source workbook")
    expert = tmp_path / "expert.docx"
    expert.write_bytes(b"expert report")
    files = {"reference_files/source.xlsx": str(source), "deliverable_files/expert.docx": str(expert)}

    with (
        patch("datasets.load_dataset", return_value=rows),
        patch("huggingface_hub.hf_hub_download", side_effect=lambda _repo, path, **_kwargs: files[path]),
        patch.object(gb, "_dataset_revision", return_value="abc123"),
    ):
        out = tmp_path / "gdpval"
        gb.build_benchmark(out_dir=out, limit=1, repair_office_files=False, register=False)
    return out


def test_rubric_and_prompt_reach_task_metadata(tmp_path):
    """The rubric reward_fn reads both off ``task.metadata``."""
    out = _build_with_rubric(tmp_path)
    task = next(t for t in BenchmarkLoader.load(str(out)).tasks if t.id == "task-one")

    assert json.loads(task.metadata["rubric_json"]) == json.loads(_REAL_RUBRIC)
    # The raw work request, not the AA instruction — that wraps the request in
    # submission boilerplate the judge should not weigh.
    assert task.metadata["prompt"] == "Review the workbook and produce a report."
    assert "finish" not in task.metadata["prompt"]


def test_the_rubric_never_reaches_the_solver(tmp_path):
    """The rubric states the expected answers, so it must stay off the sandbox.

    ``task.toml`` is host-side only and ``environment/`` becomes the image build
    context, so the rubric may live in the former but never the latter. This is
    the deliberate divergence from the cookbook, which writes
    ``tests/rubric.json`` for its in-sandbox grader — there is no in-sandbox
    grader here.
    """
    out = _build_with_rubric(tmp_path)
    task_dir = out / "task-one"

    assert "220" in (task_dir / "task.toml").read_text()  # it *is* recorded, host-side
    assert not (task_dir / "tests" / "rubric.json").exists()

    for staged in list((task_dir / "environment").rglob("*")) + list((task_dir / "tests").rglob("*")):
        if not staged.is_file():
            continue
        body = staged.read_bytes()
        assert b"The recommended sample size is 220" not in body, f"rubric leaked into {staged.relative_to(task_dir)}"


def test_a_task_with_no_rubric_still_builds(tmp_path):
    """``rubric_json`` is absent for some rows; the grader reports those ungraded."""
    out = _build(tmp_path, limit=1)
    task = next(t for t in BenchmarkLoader.load(str(out)).tasks if t.id == "task-one")
    assert json.loads(task.metadata["rubric_json"]) == []

from __future__ import annotations

import json
import os
import sys
import types
import zipfile
from pathlib import Path
from unittest.mock import patch

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
    ):
        out = tmp_path / "gdpval"
        gb.build_benchmark(out_dir=out, limit=limit, repair_office_files=False, register=False)
    return out


def test_builder_writes_rllm_harbor_layout(tmp_path):
    out = _build(tmp_path)
    task = out / "task-one"

    assert (out / "dataset.toml").exists()
    assert (task / "task.toml").exists()
    assert (task / "instruction.md").exists()
    assert (task / "environment" / "Dockerfile").exists()
    assert (task / "tests" / "test.sh").exists()
    assert (task / "tests" / "evaluate.py").exists()
    assert os.access(task / "tests" / "test.sh", os.X_OK)

    # Inputs enter the solver environment; expert outputs remain hidden in
    # tests/ until the verifier is uploaded after the agent finishes.
    assert (task / "environment" / "files" / "source.xlsx").read_bytes() == b"source workbook"
    assert (task / "tests" / "reference" / "expert.docx").read_bytes() == b"expert report"
    assert not (task / "environment" / "files" / "expert.docx").exists()


def test_task_config_uses_stirrup_and_verifier_only_judge_credentials(tmp_path):
    out = _build(tmp_path, limit=1)
    task_toml = (out / "task-one" / "task.toml").read_text()
    dataset_toml = (out / "dataset.toml").read_text()

    assert 'default_agent = "stirrup"' in dataset_toml
    assert 'script = "tests/test.sh"' in task_toml
    assert "[verifier.env]" in task_toml
    assert 'GDPVAL_JUDGE_API_KEY = "${GDPVAL_JUDGE_API_KEY}"' in task_toml
    assert 'GDPVAL_JUDGE_MODEL = "${GDPVAL_JUDGE_MODEL}"' in task_toml
    assert "[environment.env]" not in task_toml

    instruction = (out / "task-one" / "instruction.md").read_text()
    assert "- source.xlsx" in instruction
    assert "/home/user/source.xlsx" not in instruction
    assert "relative paths" in instruction


def test_builder_output_round_trips_through_benchmark_loader(tmp_path):
    out = _build(tmp_path)

    result = BenchmarkLoader.load(str(out))

    assert result.name == "gdpval"
    assert result.harness_name == "stirrup"
    assert sorted(task.id for task in result.tasks) == ["task-one", "task-without-gold"]
    by_id = {task.id: task for task in result.tasks}
    assert by_id["task-one"].metadata["reference_files"] == ["source.xlsx"]


def test_builder_respects_limit(tmp_path):
    out = _build(tmp_path, limit=1)

    task_dirs = sorted(path.name for path in out.iterdir() if path.is_dir() and not path.name.startswith("."))
    assert task_dirs == ["task-one"]


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
        package.writestr("[Content_Types].xml", "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"/>")
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
    with zipfile.ZipFile(source) as package:
        assert "word/settings.xml" not in package.namelist()


def test_dataset_catalog_entry_points_to_builder():
    catalog_path = Path(gb.__file__).parents[1] / "registry" / "datasets.json"
    catalog = json.loads(catalog_path.read_text())

    entry = catalog["datasets"]["gdpval"]
    assert entry["builder"] == "rllm.data.gdpval_builder:build_benchmark"
    assert entry["default_agent"] == "stirrup"


def test_generated_judge_normalizes_rllm_openrouter_model_prefix():
    assert 'judge_model.startswith("openrouter/")' in gb._VERIFIER_SOURCE
    assert 'judge_model = judge_model.removeprefix("openrouter/")' in gb._VERIFIER_SOURCE


def test_generated_judge_sends_normalized_model_to_openrouter(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace())
    namespace = {"__name__": "gdpval_verifier_test"}
    exec(gb._VERIFIER_SOURCE, namespace)
    candidate = tmp_path / "candidate.txt"
    reference = tmp_path / "reference.txt"
    candidate.write_text("candidate")
    reference.write_text("reference")
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"winner\\":\\"tie\\"}"}}]}'

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setenv("GDPVAL_JUDGE_API_KEY", "secret")
    monkeypatch.setenv("GDPVAL_JUDGE_MODEL", "openrouter/google/gemini-3.1-pro-preview")
    monkeypatch.setattr(namespace["urllib"].request, "urlopen", urlopen)

    namespace["judge"]({"task_id": "task", "prompt": "prompt", "rubric": "rubric"}, [candidate], [reference])

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["payload"]["model"] == "google/gemini-3.1-pro-preview"
    assert captured["timeout"] == 600

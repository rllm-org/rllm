"""Tests for :func:`rllm.eval.materialize.materialize_benchmark`.

Covers the on-disk shape produced for a catalog dataset:

- ``data/<split>.jsonl`` with one row per task
- ``instruction.md.tpl`` template
- ``dataset.toml`` with ``[dataset]`` and ``[verifier]`` sections
- MCQ category gets a ``{{question}}\\n\\n{{choices}}`` template
- VLM category extracts PIL/byte images to ``images/`` and stores paths
- Non-serialisable row values are stringified instead of crashing
"""

from __future__ import annotations

import json

import pytest
import tomllib

from rllm.eval.materialize import materialize_benchmark

# ---------------------------------------------------------------------------
# Plain text dataset
# ---------------------------------------------------------------------------


def test_materialize_creates_expected_layout(tmp_path):
    rows = [{"question": "1+1?", "ground_truth": "2"}]
    catalog_entry = {
        "instruction_field": "question",
        "metadata_fields": ["ground_truth"],
        "verifier": "math_reward_fn",
        "description": "tiny math",
        "category": "math",
    }

    bench = materialize_benchmark("tinybench", "test", rows, catalog_entry, benchmark_root=tmp_path)

    assert bench == tmp_path / "tinybench"
    assert (bench / "data" / "test.jsonl").is_file()
    assert (bench / "instruction.md.tpl").is_file()
    assert (bench / "dataset.toml").is_file()


def test_materialize_writes_jsonl_rows(tmp_path):
    rows = [
        {"question": "a?", "ground_truth": "1"},
        {"question": "b?", "ground_truth": "2"},
    ]
    bench = materialize_benchmark("ds", "test", rows, {"instruction_field": "question"}, benchmark_root=tmp_path)
    lines = (bench / "data" / "test.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["question"] == "a?"
    assert json.loads(lines[1])["ground_truth"] == "2"


def test_materialize_writes_instruction_template(tmp_path):
    bench = materialize_benchmark("ds", "test", [], {"instruction_field": "question"}, benchmark_root=tmp_path)
    tpl = (bench / "instruction.md.tpl").read_text()
    assert "{{question}}" in tpl


def test_materialize_skips_template_when_no_instruction_field(tmp_path):
    bench = materialize_benchmark("ds", "test", [], {}, benchmark_root=tmp_path)
    assert not (bench / "instruction.md.tpl").exists()


# ---------------------------------------------------------------------------
# dataset.toml shape
# ---------------------------------------------------------------------------


def test_dataset_toml_has_verifier_name(tmp_path):
    bench = materialize_benchmark("ds", "test", [], {"verifier": "math_reward_fn"}, benchmark_root=tmp_path)
    cfg = tomllib.loads((bench / "dataset.toml").read_text())
    assert cfg["verifier"]["name"] == "math_reward_fn"


def test_dataset_toml_uses_import_path_for_colon_value(tmp_path):
    bench = materialize_benchmark("ds", "test", [], {"verifier": "rllm.eval.reward_fns.math:evaluate"}, benchmark_root=tmp_path)
    cfg = tomllib.loads((bench / "dataset.toml").read_text())
    assert "import_path" in cfg["verifier"]
    assert "name" not in cfg["verifier"]


def test_dataset_toml_falls_back_to_reward_fn(tmp_path):
    """Older catalog entries declare ``reward_fn`` instead of ``verifier``."""
    bench = materialize_benchmark("ds", "test", [], {"reward_fn": "math_reward_fn"}, benchmark_root=tmp_path)
    cfg = tomllib.loads((bench / "dataset.toml").read_text())
    assert cfg["verifier"]["name"] == "math_reward_fn"


def test_dataset_toml_includes_metadata_fields(tmp_path):
    bench = materialize_benchmark(
        "ds",
        "test",
        [],
        {"instruction_field": "question", "metadata_fields": ["ground_truth", "data_source"]},
        benchmark_root=tmp_path,
    )
    cfg = tomllib.loads((bench / "dataset.toml").read_text())
    assert cfg["dataset"]["instruction_field"] == "question"
    assert cfg["dataset"]["metadata_fields"] == ["ground_truth", "data_source"]


# ---------------------------------------------------------------------------
# MCQ template
# ---------------------------------------------------------------------------


def test_mcq_template_includes_choices_block(tmp_path):
    bench = materialize_benchmark(
        "mmlu_pro_lite",
        "test",
        [],
        {"category": "mcq", "instruction_field": "question"},
        benchmark_root=tmp_path,
    )
    tpl = (bench / "instruction.md.tpl").read_text()
    assert "{{question}}" in tpl
    assert "{{choices}}" in tpl


def test_non_mcq_template_omits_choices_block(tmp_path):
    bench = materialize_benchmark("ds", "test", [], {"category": "math", "instruction_field": "question"}, benchmark_root=tmp_path)
    tpl = (bench / "instruction.md.tpl").read_text()
    assert "{{choices}}" not in tpl


# ---------------------------------------------------------------------------
# VLM image extraction
# ---------------------------------------------------------------------------


def test_vlm_extracts_pil_images(tmp_path):
    pil = pytest.importorskip("PIL.Image")
    img = pil.new("RGB", (4, 4), color=(255, 0, 0))
    rows = [{"question": "What's this?", "images": [img]}]

    bench = materialize_benchmark(
        "vlm_lite",
        "test",
        rows,
        {"category": "vlm", "instruction_field": "question"},
        benchmark_root=tmp_path,
    )

    images_dir = bench / "images"
    assert images_dir.is_dir()
    written = sorted(p.name for p in images_dir.glob("*.png"))
    assert written, "no PNGs written"

    # JSONL row references the saved path, not the PIL object
    row = json.loads((bench / "data" / "test.jsonl").read_text().splitlines()[0])
    assert isinstance(row["images"], list)
    assert all(p.startswith("images/") for p in row["images"])


def test_vlm_extracts_raw_bytes(tmp_path):
    rows = [{"question": "?", "image": b"\x89PNG\r\n\x1a\nfake"}]
    bench = materialize_benchmark("ds", "test", rows, {"category": "vlm", "instruction_field": "question"}, benchmark_root=tmp_path)
    row = json.loads((bench / "data" / "test.jsonl").read_text().splitlines()[0])
    assert row["image"].startswith("images/")
    assert (bench / row["image"]).is_file()


def test_non_vlm_does_not_create_images_dir(tmp_path):
    rows = [{"question": "?", "ground_truth": "x"}]
    bench = materialize_benchmark("ds", "test", rows, {"category": "math", "instruction_field": "question"}, benchmark_root=tmp_path)
    assert not (bench / "images").exists()


# ---------------------------------------------------------------------------
# Non-serialisable rows
# ---------------------------------------------------------------------------


def test_non_serialisable_value_stringified(tmp_path):
    class Weird:
        def __str__(self):
            return "weird-thing"

    rows = [{"question": "x?", "obj": Weird()}]
    bench = materialize_benchmark("ds", "test", rows, {"instruction_field": "question"}, benchmark_root=tmp_path)
    row = json.loads((bench / "data" / "test.jsonl").read_text().splitlines()[0])
    assert row["obj"] == "weird-thing"


def test_bytes_in_non_vlm_dataset_replaced_with_placeholder(tmp_path):
    rows = [{"question": "x?", "blob": b"hello"}]
    bench = materialize_benchmark("ds", "test", rows, {"instruction_field": "question"}, benchmark_root=tmp_path)
    row = json.loads((bench / "data" / "test.jsonl").read_text().splitlines()[0])
    assert row["blob"].startswith("<bytes:")

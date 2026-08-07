from __future__ import annotations

import hashlib
import json

import pytest

from rllm.data import mcp_atlas_builder as builder
from rllm.tasks.loader import BenchmarkLoader


def _row(idx: int) -> dict:
    return {
        "TASK": f"task-{idx:03d}",
        "PROMPT": f"Prompt {idx}",
        "ENABLED_TOOLS": json.dumps(["filesystem_read_file", "git_git_status"]),
        "GTFA_CLAIMS": json.dumps([f"Claim {idx}A", f"Claim {idx}B"]),
        "TRAJECTORY": f"raw trajectory {idx}",
    }


def test_normalize_row_preserves_order_and_source_fields():
    row = _row(7)
    normalized = builder.normalize_row(row)

    assert normalized["id"] == "task-007"
    assert normalized["PROMPT"] == "Prompt 7"
    assert normalized["ENABLED_TOOLS"] == ["filesystem_read_file", "git_git_status"]
    assert normalized["GTFA_CLAIMS"] == ["Claim 7A", "Claim 7B"]
    assert normalized["TRAJECTORY"] == "raw trajectory 7"


def test_normalize_row_handles_pinned_claim_with_literal_newline():
    row = _row(22)
    row["GTFA_CLAIMS"] = "['First line\\nSecond line\n']"

    normalized = builder.normalize_row(row)

    assert normalized["GTFA_CLAIMS"] == ["First line\nSecond line"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("ENABLED_TOOLS", "not a list", "malformed ENABLED_TOOLS"),
        ("GTFA_CLAIMS", "[]", "GTFA_CLAIMS is empty"),
    ],
)
def test_normalize_row_rejects_malformed_lists(field, value, match):
    row = _row(0)
    row[field] = value
    with pytest.raises(ValueError, match=match):
        builder.normalize_row(row)


def test_build_validates_pin_and_round_trips(monkeypatch, tmp_path):
    parquet = tmp_path / builder.PARQUET_FILENAME
    parquet.write_bytes(b"pinned parquet fixture")
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    rows = [_row(idx) for idx in range(builder.PUBLIC_TASK_COUNT)]
    download_calls = []

    def fake_download(**kwargs):
        download_calls.append(kwargs)
        return str(parquet)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: rows)
    monkeypatch.setattr(builder, "PARQUET_SHA256", digest)

    out = builder.build_benchmark(out_dir=tmp_path / "benchmark", register=False)

    assert download_calls == [
        {
            "repo_id": builder.REPO_ID,
            "filename": builder.PARQUET_FILENAME,
            "repo_type": "dataset",
            "revision": builder.REVISION,
        }
    ]
    loaded = BenchmarkLoader.load(str(out))
    assert loaded.split == "public"
    assert loaded.harness_name == "mcp-atlas"
    assert len(loaded.tasks) == builder.PUBLIC_TASK_COUNT
    assert loaded.tasks[0].instruction == "Prompt 0"
    assert loaded.tasks[0].metadata["GTFA_CLAIMS"] == ["Claim 0A", "Claim 0B"]
    manifest = json.loads((out / "source_manifest.json").read_text())
    assert manifest["revision"] == builder.REVISION
    assert manifest["parquet_sha256"] == digest


def test_build_rejects_checksum_mismatch(monkeypatch, tmp_path):
    parquet = tmp_path / builder.PARQUET_FILENAME
    parquet.write_bytes(b"wrong")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(parquet))

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        builder.build_benchmark(out_dir=tmp_path / "benchmark", register=False)


def test_build_rejects_duplicate_task_ids(monkeypatch, tmp_path):
    parquet = tmp_path / builder.PARQUET_FILENAME
    parquet.write_bytes(b"fixture")
    rows = [_row(idx) for idx in range(builder.PUBLIC_TASK_COUNT)]
    rows[-1]["TASK"] = rows[0]["TASK"]
    monkeypatch.setattr(builder, "PARQUET_SHA256", hashlib.sha256(parquet.read_bytes()).hexdigest())
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(parquet))
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: rows)

    with pytest.raises(RuntimeError, match="duplicate TASK"):
        builder.build_benchmark(out_dir=tmp_path / "benchmark", register=False)

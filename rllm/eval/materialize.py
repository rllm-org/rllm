"""Materialize a catalog dataset into a self-contained benchmark directory.

After materialization, ``~/.rllm/datasets/<name>/`` contains everything
needed to run the dataset through the standard :class:`rllm.runner.Runner`:

::

    ~/.rllm/datasets/<name>/
    ├── dataset.toml          # name, fields, verifier reference
    ├── instruction.md.tpl    # template, e.g. "Solve: {{question}}"
    └── data/
        └── <split>.jsonl     # one row per task

The existing ``<split>.parquet`` (used by verl-based training) is left
alongside untouched.

Catalog entries gain three optional fields:

::

    "instruction_field": "question",
    "metadata_fields": ["answer", "ground_truth"],
    "verifier": "math_reward_fn"      // registered name, OR "module:fn" import path

If ``instruction_field`` is omitted, the materialize step still writes the
data file but no instruction template (the user can author one later).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def materialize_benchmark(
    name: str,
    split: str,
    rows: list[dict[str, Any]],
    catalog_entry: dict,
    benchmark_root: str | Path | None = None,
) -> Path:
    """Write a benchmark directory under ``benchmark_root/<name>/``.

    Args:
        name: Dataset name (also the directory name).
        split: Split being materialized (e.g. "train", "test").
        rows: List of row dicts (already transformed/field-mapped).
        catalog_entry: The full catalog entry from datasets.json. Reads
            ``instruction_field``, ``metadata_fields``, ``verifier``,
            ``description``, ``category``.
        benchmark_root: Override storage root. Default
            ``~/.rllm/datasets``.

    Returns:
        Path to the benchmark directory.
    """
    if benchmark_root is None:
        benchmark_root = os.path.join(
            os.environ.get("RLLM_HOME", os.path.expanduser("~/.rllm")),
            "datasets",
        )
    bench_dir = Path(benchmark_root) / name
    bench_dir.mkdir(parents=True, exist_ok=True)

    _write_data_jsonl(bench_dir, split, rows, catalog_entry)
    _write_instruction_template(bench_dir, catalog_entry)
    _write_dataset_toml(bench_dir, name, catalog_entry)

    logger.info("Materialized %s/%s → %s (%d rows)", name, split, bench_dir, len(rows))
    return bench_dir


def _write_data_jsonl(bench_dir: Path, split: str, rows: list[dict], catalog_entry: dict) -> None:
    """Write rows to data/<split>.jsonl. Strips binary columns (images)."""
    data_dir = bench_dir / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / f"{split}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            cleaned = _strip_non_serializable(row)
            f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")


def _write_instruction_template(bench_dir: Path, catalog_entry: dict) -> None:
    """Write instruction.md.tpl using ``instruction_field`` if set."""
    field = catalog_entry.get("instruction_field")
    if not field:
        return
    tpl_path = bench_dir / "instruction.md.tpl"
    tpl_path.write_text("{{" + field + "}}\n", encoding="utf-8")


def _write_dataset_toml(bench_dir: Path, name: str, catalog_entry: dict) -> None:
    """Write dataset.toml with name, fields, and verifier reference."""
    lines = [
        "[dataset]",
        f'name = "{name}"',
    ]
    if catalog_entry.get("description"):
        lines.append(f'description = "{_escape_toml_string(catalog_entry["description"])}"')
    if catalog_entry.get("category"):
        lines.append(f'category = "{catalog_entry["category"]}"')
    if catalog_entry.get("instruction_field"):
        lines.append(f'instruction_field = "{catalog_entry["instruction_field"]}"')
    if catalog_entry.get("metadata_fields"):
        fields = ", ".join(f'"{f}"' for f in catalog_entry["metadata_fields"])
        lines.append(f"metadata_fields = [{fields}]")

    verifier_ref = catalog_entry.get("verifier") or catalog_entry.get("reward_fn")
    if verifier_ref:
        lines.append("")
        lines.append("[verifier]")
        if ":" in verifier_ref:
            lines.append(f'import_path = "{verifier_ref}"')
        else:
            lines.append(f'name = "{verifier_ref}"')

    (bench_dir / "dataset.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _strip_non_serializable(row: dict) -> dict:
    """Replace bytes/non-JSON values with placeholders so jsonl works.

    Binary columns (image bytes, audio bytes) are dropped; their parquet
    sibling still has them.
    """
    out: dict = {}
    for k, v in row.items():
        if isinstance(v, bytes):
            out[k] = f"<bytes:{len(v)}>"
        elif isinstance(v, list) and v and isinstance(v[0], bytes):
            out[k] = [f"<bytes:{len(b)}>" if isinstance(b, bytes) else b for b in v]
        else:
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def _escape_toml_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

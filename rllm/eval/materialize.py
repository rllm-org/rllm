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
    """Write rows to data/<split>.jsonl.

    For VLM benchmarks (``category == "vlm"``) any image-typed values
    (PIL Image, bytes, list of either) are extracted to ``images/`` as
    PNG files and replaced with their relative paths so the jsonl stays
    serialisable. The loader reconstructs multimodal content blocks at
    read time.
    """
    data_dir = bench_dir / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / f"{split}.jsonl"

    is_vlm = catalog_entry.get("category") == "vlm"
    images_dir = bench_dir / "images" if is_vlm else None
    if images_dir is not None:
        images_dir.mkdir(exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            if is_vlm:
                row = _extract_images_from_row(row, idx, images_dir)
            cleaned = _strip_non_serializable(row)
            f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")


def _extract_images_from_row(row: dict, row_idx: int, images_dir: Path) -> dict:
    """Replace image-typed values with their saved filename (relative path).

    Looks at every column value: PIL Images, raw bytes, and lists of
    either. Each image is written under ``images/<idx>_<col>[_n].png``;
    the column value becomes the path string (or list of paths).
    """
    out: dict = {}
    for col, value in row.items():
        out[col] = _replace_images(value, row_idx, col, images_dir)
    return out


def _replace_images(value, row_idx: int, col: str, images_dir: Path):
    """If *value* is image-like, save it and return the relative path."""
    if _is_pil_image(value) or isinstance(value, bytes):
        path = images_dir / f"{row_idx}_{col}.png"
        _save_image(value, path)
        return f"images/{path.name}"
    if isinstance(value, list):
        replaced = []
        any_image = False
        for n, item in enumerate(value):
            if _is_pil_image(item) or isinstance(item, bytes):
                path = images_dir / f"{row_idx}_{col}_{n}.png"
                _save_image(item, path)
                replaced.append(f"images/{path.name}")
                any_image = True
            else:
                replaced.append(item)
        return replaced if any_image else value
    return value


def _is_pil_image(value) -> bool:
    try:
        from PIL.Image import Image as PILImage
    except ImportError:
        return False
    return isinstance(value, PILImage)


def _save_image(value, path: Path) -> None:
    if isinstance(value, bytes):
        path.write_bytes(value)
        return
    # PIL Image
    img = value
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    img.save(path, format="PNG")


def _write_instruction_template(bench_dir: Path, catalog_entry: dict) -> None:
    """Write ``instruction.md.tpl`` using ``instruction_field`` if set.

    For MCQ datasets (``category == "mcq"``), also append a
    ``{{choices}}`` block so the rendered prompt includes the lettered
    options that the loader formats from the row's ``choices`` list.
    """
    field = catalog_entry.get("instruction_field")
    if not field:
        return
    tpl_path = bench_dir / "instruction.md.tpl"
    if catalog_entry.get("category") == "mcq":
        body = "{{" + field + "}}\n\n{{choices}}\n"
    else:
        body = "{{" + field + "}}\n"
    tpl_path.write_text(body, encoding="utf-8")


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

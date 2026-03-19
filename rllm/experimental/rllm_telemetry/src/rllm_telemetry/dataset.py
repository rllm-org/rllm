"""Dataset loading for experiments.

Provides :func:`load_dataset` which loads test cases from files, the rllm
dataset registry, or inline data — and maps them to the ``{"input": ...,
"expected": ...}`` format that :func:`~rllm_telemetry.eval.Eval` expects.

Supports all formats the rllm ``Dataset`` class handles: JSON, JSONL, CSV,
Parquet, and Arrow.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("rllm_telemetry.dataset")

# Default field mapping: rllm convention → Eval convention
_DEFAULT_FIELD_MAP: dict[str, str] = {
    "question": "input",
    "ground_truth": "expected",
}


def _apply_field_map(
    row: dict[str, Any],
    field_map: dict[str, str],
) -> dict[str, Any]:
    """Remap fields in a row according to ``field_map``.

    Mapped fields are renamed; unmapped fields are kept as-is.
    """
    mapped: dict[str, Any] = {}
    for key, value in row.items():
        new_key = field_map.get(key, key)
        mapped[new_key] = value
    # Ensure 'input' exists — if not, try first string value
    if "input" not in mapped:
        for v in mapped.values():
            if isinstance(v, str):
                mapped["input"] = v
                break
    return mapped


def _load_file(path: str) -> list[dict[str, Any]]:
    """Load data from a file, supporting JSON, JSONL, CSV, Parquet, Arrow."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # Handle {"data": [...]} or {"examples": [...]} wrappers
            for key in ("data", "examples", "test_cases", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        return data

    if ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if ext == ".csv":
        import pandas as pd

        return pd.read_csv(path).to_dict("records")

    if ext == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")

    if ext == ".arrow":
        # Use rllm's Dataset if available, otherwise pyarrow directly
        try:
            from rllm.data.dataset import Dataset as RllmDataset

            ds = RllmDataset.load_data(path)
            return ds.get_data()
        except ImportError:
            import pyarrow.ipc as ipc

            with open(path, "rb") as f:
                reader = ipc.open_file(f)
                table = reader.read_all()
            return table.to_pylist()

    raise ValueError(f"Unsupported file format: {ext}. Supported: .json, .jsonl, .csv, .parquet, .arrow")


def load_dataset(
    source: str,
    *,
    split: str | None = None,
    field_map: dict[str, str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load a dataset and return it as a list of Eval-compatible dicts.

    Args:
        source: One of:
            - A file path (``".json"``, ``".jsonl"``, ``".csv"``,
              ``".parquet"``, ``".arrow"``)
            - A registered rllm dataset name (e.g. ``"gsm8k"``,
              ``"math500"``)
        split: Dataset split (required for registry datasets, e.g.
            ``"test"``). Ignored for file paths.
        field_map: Custom mapping of ``{source_field: target_field}``.
            Defaults to ``{"question": "input", "ground_truth": "expected"}``
            which matches the standard rllm dataset convention.
        limit: Optional max number of examples to return.

    Returns:
        List of dicts, each with at least an ``"input"`` key and optionally
        ``"expected"`` plus any extra fields from the source data.
    """
    fmap = {**_DEFAULT_FIELD_MAP, **(field_map or {})}

    # Try as file path first
    if os.path.exists(source):
        raw = _load_file(source)
    else:
        # Try rllm DatasetRegistry
        raw = _load_from_registry(source, split=split)

    # Apply field mapping
    mapped = [_apply_field_map(row, fmap) for row in raw]

    if limit is not None:
        mapped = mapped[:limit]

    return mapped


def _load_from_registry(name: str, *, split: str | None = None) -> list[dict[str, Any]]:
    """Load a dataset from the rllm DatasetRegistry."""
    try:
        from rllm.data.dataset import DatasetRegistry
    except ImportError as err:
        raise ImportError(f"Cannot load dataset '{name}' from registry: rllm is not installed. Provide a file path instead.") from err

    if split is None:
        # Try common eval splits
        for try_split in ("test", "validation", "eval", "train"):
            if DatasetRegistry.dataset_exists(name, try_split):
                split = try_split
                break
        if split is None:
            raise ValueError(f"Dataset '{name}' not found in registry. Pull it first with: rllm dataset pull " + name)

    ds = DatasetRegistry.load_dataset(name, split)
    return ds.get_data()

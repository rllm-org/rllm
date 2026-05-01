"""Read-side helpers for the Datasets panel.

Two on-disk sources merge into the panel's view of "available datasets":

* ``rllm/registry/datasets.json``        — curated descriptive registry
  (name, source, category, splits, transform, reward_fn, …). The
  source of truth for what datasets *exist* in the rLLM ecosystem.

* ``rllm/registry/dataset_registry.json``— a per-machine map of which
  splits have been downloaded into local parquet caches under
  ``rllm/data/datasets/<name>/<split>.parquet``. Used to mark a
  dataset as locally browsable and to read entries from.

Entry reading uses ``polars`` (already a project dep) for cheap slicing
without loading the full parquet into memory.
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _registry_dir() -> Path:
    # rllm/console/panels/datasets/loader.py → ../../../registry/
    return Path(__file__).resolve().parents[3] / "registry"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("datasets-panel: could not read %s", path)
        return {}


@lru_cache(maxsize=1)
def _load_descriptive() -> dict[str, dict[str, Any]]:
    """``{name: entry}`` from ``datasets.json`` (the curated registry)."""
    body = _load_json(_registry_dir() / "datasets.json")
    return body.get("datasets", {}) if isinstance(body, dict) else {}


@lru_cache(maxsize=1)
def _load_local_paths() -> dict[str, dict[str, str]]:
    """``{name: {split: path}}`` from ``dataset_registry.json``."""
    body = _load_json(_registry_dir() / "dataset_registry.json")
    return body if isinstance(body, dict) else {}


def list_datasets() -> list[dict[str, Any]]:
    """Card-friendly summaries for every registered dataset.

    Each row carries the descriptive metadata plus a ``local_splits``
    list (subset of ``splits`` that have a local parquet) so the card
    can show a "browsable / not downloaded" state at a glance.
    """
    desc = _load_descriptive()
    local = _load_local_paths()
    out: list[dict[str, Any]] = []
    for name, entry in desc.items():
        local_for_name = local.get(name, {})
        out.append(
            {
                "name": name,
                "description": entry.get("description") or "",
                "source": entry.get("source") or "",
                "category": entry.get("category") or "uncategorized",
                "splits": entry.get("splits") or [],
                "default_agent": entry.get("default_agent"),
                "reward_fn": entry.get("reward_fn"),
                "eval_split": entry.get("eval_split"),
                "instruction_field": entry.get("instruction_field"),
                "transform": entry.get("transform"),
                "local_splits": sorted(local_for_name.keys()),
                "is_local": bool(local_for_name),
            }
        )
    out.sort(key=lambda r: (r["category"], r["name"]))
    return out


def categories() -> list[str]:
    """Sorted unique category names; "uncategorized" at the end."""
    cats = {(e.get("category") or "uncategorized") for e in _load_descriptive().values()}
    return sorted(c for c in cats if c != "uncategorized") + (["uncategorized"] if "uncategorized" in cats else [])


def get_dataset(name: str) -> dict[str, Any] | None:
    """Full detail for one dataset, including per-split entry counts."""
    desc = _load_descriptive().get(name)
    if not desc:
        return None
    local_for_name = _load_local_paths().get(name, {})

    split_info: list[dict[str, Any]] = []
    for split in desc.get("splits") or []:
        path = local_for_name.get(split)
        info: dict[str, Any] = {
            "name": split,
            "is_local": path is not None,
            "path": path,
            "n_rows": None,
            "schema": None,
        }
        if path:
            try:
                meta = _parquet_metadata(Path(path))
                info["n_rows"] = meta["n_rows"]
                info["schema"] = meta["schema"]
            except Exception:
                logger.exception("datasets-panel: could not read parquet meta %s", path)
        split_info.append(info)

    return {
        **desc,
        "name": name,
        "splits_detail": split_info,
        "local_splits": sorted(local_for_name.keys()),
        "is_local": bool(local_for_name),
    }


def _parquet_metadata(path: Path) -> dict[str, Any]:
    """Cheap row-count + schema read without loading the data."""
    import polars as pl

    lf = pl.scan_parquet(path)
    schema = {k: str(v) for k, v in dict(lf.collect_schema()).items()}
    n_rows = lf.select(pl.len()).collect().item()
    return {"n_rows": int(n_rows), "schema": schema}


def get_entries(
    name: str,
    *,
    split: str,
    offset: int = 0,
    limit: int = 25,
) -> dict[str, Any]:
    """Slice rows from the local parquet for ``(name, split)``.

    Returns ``{rows, total, offset, limit, columns}``. Rows are dicts
    keyed by column name; values are JSON-serialisable (polars does the
    coercion via ``to_dicts()``).
    """
    if limit < 1 or limit > 200:
        raise ValueError("limit must be in [1, 200]")
    if offset < 0:
        raise ValueError("offset must be >= 0")

    path_str = _load_local_paths().get(name, {}).get(split)
    if not path_str:
        raise FileNotFoundError(f"no local parquet for {name}/{split}")

    import polars as pl

    lf = pl.scan_parquet(Path(path_str))
    columns = list(dict(lf.collect_schema()).keys())
    total = int(lf.select(pl.len()).collect().item())

    if offset >= total:
        rows: list[dict[str, Any]] = []
    else:
        df = lf.slice(offset, limit).collect()
        rows = df.to_dicts()

    return {
        "rows": rows,
        "total": total,
        "offset": offset,
        "limit": limit,
        "columns": columns,
        "n_pages": math.ceil(total / limit) if total else 0,
    }

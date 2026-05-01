"""Read-side helpers for the Datasets panel.

Three on-disk sources contribute to the panel's view of "available
datasets":

* ``<RLLM_HOME>/datasets/registry.json``  — the **user-level v2
  registry** that ``rllm dataset pull`` writes to. Source of truth
  for what's locally available. Schema::

      {"version": 2, "datasets": {<name>: {
          "metadata": {"source", "description", "category", ...},
          "splits": {<split>: {
              "path": "<name>/<split>.parquet",   # relative to datasets/
              "num_examples": int,
              "fields": [str, ...],
          }}
      }}}

* ``rllm/registry/datasets.json``         — the curated descriptive
  registry shipped with the repo (transform path, default agent,
  reward fn, etc.). Augments user-pulled datasets with the rLLM-side
  metadata that the user registry doesn't carry.

* ``rllm/registry/dataset_registry.json`` — legacy v1 paths for the
  small set of datasets that were bundled in the repo before the
  user-level registry existed. Used as a last-ditch fallback for
  resolving local parquet paths.

Reads are *not* cached: the user pulls datasets at runtime via
``rllm dataset pull`` and we want them to appear in the UI without
restarting ``rllm view``. The files are small (<100KB), so the cost
of re-reading is negligible.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _rllm_home() -> Path:
    return Path(os.path.expanduser(os.environ.get("RLLM_HOME", "~/.rllm")))


def _user_datasets_root() -> Path:
    return _rllm_home() / "datasets"


def _user_registry_path() -> Path:
    return _user_datasets_root() / "registry.json"


def _project_registry_dir() -> Path:
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


# ---------------------------------------------------------------------------
# Registry readers
# ---------------------------------------------------------------------------


def _load_descriptive() -> dict[str, dict[str, Any]]:
    """``{name: entry}`` from the curated repo-shipped registry."""
    body = _load_json(_project_registry_dir() / "datasets.json")
    return body.get("datasets", {}) if isinstance(body, dict) else {}


def _load_user_registry() -> dict[str, dict[str, Any]]:
    """``{name: entry}`` from the user-level v2 registry.

    Returns ``{}`` when the file is missing (fresh install). Older v1
    files (a flat ``{name: {split: path}}`` map) are coerced into the
    v2 shape so downstream code only deals with one schema.
    """
    body = _load_json(_user_registry_path())
    if not isinstance(body, dict):
        return {}

    if body.get("version") == 2 and isinstance(body.get("datasets"), dict):
        return body["datasets"]

    # Coerce v1 → v2-shape so the rest of the loader treats it uniformly.
    coerced: dict[str, dict[str, Any]] = {}
    for name, entry in body.items():
        if not isinstance(entry, dict):
            continue
        coerced[name] = {
            "metadata": {},
            "splits": {split: {"path": path, "num_examples": None, "fields": []} for split, path in entry.items() if isinstance(path, str)},
        }
    return coerced


def _load_legacy_paths() -> dict[str, dict[str, str]]:
    """``{name: {split: absolute_path}}`` from the legacy in-repo file."""
    body = _load_json(_project_registry_dir() / "dataset_registry.json")
    return body if isinstance(body, dict) else {}


def _resolve_user_split_path(rel_path: str) -> Path:
    """User-registry paths are relative to ``<RLLM_HOME>/datasets/``."""
    return _user_datasets_root() / rel_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_datasets() -> list[dict[str, Any]]:
    """Card-friendly summaries — union of descriptive + user registries.

    Datasets appear if they're known by *either* registry. Descriptive
    metadata wins for shared keys; user-registry metadata fills in for
    pulled-only datasets.
    """
    desc = _load_descriptive()
    user = _load_user_registry()
    legacy = _load_legacy_paths()

    out: list[dict[str, Any]] = []
    for name in sorted(set(desc) | set(user)):
        d_entry = desc.get(name) or {}
        u_entry = user.get(name) or {}
        u_meta = u_entry.get("metadata") or {}

        description = d_entry.get("description") or u_meta.get("description") or ""
        source = d_entry.get("source") or u_meta.get("source") or ""
        category = d_entry.get("category") or u_meta.get("category") or "uncategorized"

        # Splits: union of (descriptive splits, user splits, legacy splits).
        # Descriptive may declare splits that aren't downloaded yet; user
        # registry may have splits the descriptive entry didn't anticipate.
        all_splits = set(d_entry.get("splits") or [])
        all_splits |= set((u_entry.get("splits") or {}).keys())
        all_splits |= set(legacy.get(name, {}).keys())

        local_splits = sorted(_local_split_names(name, u_entry, legacy.get(name, {})))

        out.append(
            {
                "name": name,
                "description": description,
                "source": source,
                "category": category,
                "splits": sorted(all_splits),
                "default_agent": d_entry.get("default_agent"),
                "reward_fn": d_entry.get("reward_fn"),
                "eval_split": d_entry.get("eval_split"),
                "instruction_field": d_entry.get("instruction_field"),
                "transform": d_entry.get("transform"),
                "local_splits": local_splits,
                "is_local": bool(local_splits),
            }
        )
    out.sort(key=lambda r: (r["category"], r["name"]))
    return out


def categories() -> list[str]:
    """Sorted unique category names; "uncategorized" at the end."""
    cats: set[str] = set()
    for entry in _load_descriptive().values():
        cats.add(entry.get("category") or "uncategorized")
    for entry in _load_user_registry().values():
        meta = entry.get("metadata") or {}
        cats.add(meta.get("category") or "uncategorized")
    return sorted(c for c in cats if c != "uncategorized") + (["uncategorized"] if "uncategorized" in cats else [])


def get_dataset(name: str) -> dict[str, Any] | None:
    """Full detail for one dataset — merged metadata + per-split info."""
    desc = _load_descriptive().get(name) or {}
    user_entry = _load_user_registry().get(name) or {}
    legacy = _load_legacy_paths().get(name, {})

    if not desc and not user_entry:
        return None

    user_meta = user_entry.get("metadata") or {}
    user_splits = user_entry.get("splits") or {}

    declared_splits = list(desc.get("splits") or [])
    extra = [s for s in user_splits if s not in declared_splits]
    extra += [s for s in legacy if s not in declared_splits and s not in extra]
    all_splits = declared_splits + extra

    split_info: list[dict[str, Any]] = []
    for split in all_splits:
        info = _split_info(name, split, user_splits.get(split), legacy.get(split))
        split_info.append(info)

    local_splits = sorted(_local_split_names(name, user_entry, legacy))

    return {
        "name": name,
        "description": desc.get("description") or user_meta.get("description") or "",
        "source": desc.get("source") or user_meta.get("source") or "",
        "category": desc.get("category") or user_meta.get("category") or "uncategorized",
        "splits": all_splits,
        "default_agent": desc.get("default_agent"),
        "reward_fn": desc.get("reward_fn"),
        "eval_split": desc.get("eval_split"),
        "instruction_field": desc.get("instruction_field"),
        "transform": desc.get("transform"),
        "splits_detail": split_info,
        "local_splits": local_splits,
        "is_local": bool(local_splits),
    }


def get_entries(
    name: str,
    *,
    split: str,
    offset: int = 0,
    limit: int = 25,
) -> dict[str, Any]:
    """Slice rows from the local parquet for ``(name, split)``.

    Resolves the path via the user registry first, then falls back to
    the legacy in-repo registry. Raises ``FileNotFoundError`` when no
    local copy exists.
    """
    if limit < 1 or limit > 200:
        raise ValueError("limit must be in [1, 200]")
    if offset < 0:
        raise ValueError("offset must be >= 0")

    path = _resolve_local_path(name, split)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"no local parquet for {name}/{split}")

    import polars as pl

    lf = pl.scan_parquet(path)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_local_path(name: str, split: str) -> Path | None:
    """Return the on-disk parquet path for ``(name, split)`` or None."""
    user = _load_user_registry().get(name) or {}
    user_split = (user.get("splits") or {}).get(split)
    if isinstance(user_split, dict) and user_split.get("path"):
        candidate = _resolve_user_split_path(user_split["path"])
        if candidate.is_file():
            return candidate

    legacy = _load_legacy_paths().get(name, {}).get(split)
    if isinstance(legacy, str):
        candidate = Path(legacy)
        if candidate.is_file():
            return candidate

    return None


def _local_split_names(
    name: str,
    user_entry: dict[str, Any],
    legacy_for_name: dict[str, str],
) -> set[str]:
    """Splits that are actually readable from disk right now."""
    out: set[str] = set()
    for split, info in (user_entry.get("splits") or {}).items():
        if not isinstance(info, dict):
            continue
        rel = info.get("path")
        if isinstance(rel, str) and _resolve_user_split_path(rel).is_file():
            out.add(split)
    for split, abspath in legacy_for_name.items():
        if isinstance(abspath, str) and Path(abspath).is_file():
            out.add(split)
    return out


def _split_info(
    name: str,
    split: str,
    user_split: dict[str, Any] | None,
    legacy_path: str | None,
) -> dict[str, Any]:
    """Per-split metadata for the detail view.

    Uses the user registry's ``num_examples`` + ``fields`` when
    available (free — already in the file). Falls back to a polars
    scan only when needed (legacy path, or pulled file with no counts).
    """
    info: dict[str, Any] = {
        "name": split,
        "is_local": False,
        "path": None,
        "n_rows": None,
        "schema": None,
    }

    resolved_path: Path | None = None
    if user_split and isinstance(user_split, dict):
        rel = user_split.get("path")
        if isinstance(rel, str):
            cand = _resolve_user_split_path(rel)
            if cand.is_file():
                resolved_path = cand
                info["n_rows"] = user_split.get("num_examples")
                fields = user_split.get("fields") or []
                if fields:
                    info["schema"] = {f: "?" for f in fields}

    if resolved_path is None and legacy_path:
        cand = Path(legacy_path)
        if cand.is_file():
            resolved_path = cand

    if resolved_path is not None:
        info["is_local"] = True
        info["path"] = str(resolved_path)
        # Backfill counts/schema via polars only when the registry
        # didn't carry them (legacy entries, partial v2 records).
        if info["n_rows"] is None or info["schema"] is None:
            try:
                meta = _parquet_metadata(resolved_path)
                if info["n_rows"] is None:
                    info["n_rows"] = meta["n_rows"]
                if info["schema"] is None:
                    info["schema"] = meta["schema"]
            except Exception:
                logger.exception(
                    "datasets-panel: could not backfill parquet meta %s",
                    resolved_path,
                )

    return info


def _parquet_metadata(path: Path) -> dict[str, Any]:
    """Cheap row-count + schema read without loading the data."""
    import polars as pl

    lf = pl.scan_parquet(path)
    schema = {k: str(v) for k, v in dict(lf.collect_schema()).items()}
    n_rows = lf.select(pl.len()).collect().item()
    return {"n_rows": int(n_rows), "schema": schema}

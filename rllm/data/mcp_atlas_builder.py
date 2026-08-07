"""Materialize the public MCP-Atlas dataset as an rLLM benchmark.

The public release contains 500 rows in the Hugging Face ``train`` split.
rLLM deliberately exposes those rows as ``public`` so callers do not confuse
the release with MCP-Atlas' private 500-task leaderboard half.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ID = "ScaleAI/MCP-Atlas"
REVISION = "b5bcde2"
PARQUET_FILENAME = "MCP-Atlas.parquet"
PARQUET_SHA256 = "2d7bc052f14cbcb3b8294293481053f7111d256f9c9deaa96f3ff632d19958d0"
PUBLIC_TASK_COUNT = 500
VERIFIER_NAME = "mcp_atlas_claims"
REQUIRED_COLUMNS = ("TASK", "PROMPT", "ENABLED_TOOLS", "GTFA_CLAIMS", "TRAJECTORY")


def _parse_list(value: Any, *, field: str, task_id: str) -> list[Any]:
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                # The pinned parquet contains one otherwise-valid Python-list
                # GTFA_CLAIMS value with a literal newline immediately before
                # its closing quote. Python string literals cannot contain an
                # unescaped newline, so retry after escaping physical line
                # breaks. This preserves the intended single claim and avoids
                # the official scorer's lossy line-splitting fallback.
                try:
                    parsed = ast.literal_eval(value.replace("\r\n", "\\n").replace("\r", "\\n").replace("\n", "\\n"))
                except (SyntaxError, ValueError) as tolerant_exc:
                    raise ValueError(f"MCP-Atlas task {task_id}: malformed {field}") from tolerant_exc
    else:
        parsed = []
    if not isinstance(parsed, list):
        raise ValueError(f"MCP-Atlas task {task_id}: {field} must be a list")
    return parsed


def _tool_names(value: Any, *, task_id: str) -> list[str]:
    items = _parse_list(value, field="ENABLED_TOOLS", task_id=task_id)
    names: list[str] = []
    for item in items:
        if isinstance(item, str) and item:
            names.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
        else:
            raise ValueError(f"MCP-Atlas task {task_id}: invalid ENABLED_TOOLS entry {item!r}")
    if not names:
        raise ValueError(f"MCP-Atlas task {task_id}: ENABLED_TOOLS is empty")
    return names


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one official dataset row without reordering lists."""
    missing = [name for name in REQUIRED_COLUMNS if name not in row]
    if missing:
        raise ValueError(f"MCP-Atlas row missing required columns: {', '.join(missing)}")
    task_id = str(row["TASK"]).strip()
    prompt = str(row["PROMPT"]).strip()
    if not task_id or not prompt:
        raise ValueError("MCP-Atlas TASK and PROMPT must be non-empty")
    claims = [str(claim).strip() for claim in _parse_list(row["GTFA_CLAIMS"], field="GTFA_CLAIMS", task_id=task_id)]
    if not claims or any(not claim for claim in claims):
        raise ValueError(f"MCP-Atlas task {task_id}: GTFA_CLAIMS is empty")
    tools = _tool_names(row["ENABLED_TOOLS"], task_id=task_id)
    return {
        "id": task_id,
        "TASK": task_id,
        "PROMPT": prompt,
        "ENABLED_TOOLS": tools,
        "GTFA_CLAIMS": claims,
        # Preserve the official diagnostic trajectory exactly as published.
        "TRAJECTORY": row["TRAJECTORY"],
    }


def _write_dataset(out: Path, rows: list[dict[str, Any]], description: str) -> None:
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "public.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    dataset_toml = "\n".join(
        [
            "[dataset]",
            'name = "mcp_atlas"',
            'type = "simple"',
            f"description = {json.dumps(description)}",
            'category = "agentic"',
            'split = "public"',
            'instruction_field = "PROMPT"',
            'default_agent = "mcp-atlas"',
            'metadata_fields = ["TASK", "PROMPT", "ENABLED_TOOLS", "GTFA_CLAIMS", "TRAJECTORY"]',
            "",
            "[verifier]",
            f'name = "{VERIFIER_NAME}"',
            "",
        ]
    )
    (out / "dataset.toml").write_text(dataset_toml, encoding="utf-8")


def build_benchmark(
    *,
    name: str = "mcp_atlas",
    split: str = "public",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    limit: int | None = None,
    clean: bool = False,
    register: bool = True,
) -> Path:
    """Download, validate, materialize, and optionally register MCP-Atlas."""
    if split != "public":
        raise ValueError("MCP-Atlas public release only supports split='public'")

    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    parquet_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=PARQUET_FILENAME,
            repo_type="dataset",
            revision=REVISION,
        )
    )
    digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    if digest != PARQUET_SHA256:
        raise RuntimeError(f"MCP-Atlas parquet checksum mismatch: expected {PARQUET_SHA256}, got {digest}")

    dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
    raw_rows = [dict(row) for row in dataset]
    if len(raw_rows) != PUBLIC_TASK_COUNT:
        raise RuntimeError(f"MCP-Atlas public revision must contain {PUBLIC_TASK_COUNT} tasks, got {len(raw_rows)}")
    rows = [normalize_row(row) for row in raw_rows]
    ids = [row["TASK"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("MCP-Atlas public revision contains duplicate TASK identifiers")
    if limit is not None:
        rows = rows[:limit]

    out = Path(out_dir).expanduser()
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    description = (catalog_entry or {}).get("description") or "MCP-Atlas public 500-task tool-use benchmark"
    _write_dataset(out, rows, description)

    manifest = {
        "source": REPO_ID,
        "revision": REVISION,
        "parquet_sha256": PARQUET_SHA256,
        "public_task_count": PUBLIC_TASK_COUNT,
        "materialized_task_count": len(rows),
    }
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if register:
        from rllm.data import DatasetRegistry

        DatasetRegistry.register_dataset(
            name=name,
            data=rows,
            split="public",
            source=REPO_ID,
            description=description,
            category=(catalog_entry or {}).get("category", "agentic"),
        )
    logger.info("Materialized MCP-Atlas public split (%d tasks) at %s", len(rows), out)
    return out


__all__ = [
    "PARQUET_SHA256",
    "PUBLIC_TASK_COUNT",
    "REPO_ID",
    "REVISION",
    "build_benchmark",
    "normalize_row",
]

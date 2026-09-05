"""Build official Harbor CyberGym Level 1 tasks from ``sunblaze-ucb/cybergym``.

``rllm dataset pull cybergym`` downloads only ``tasks.json`` (~2 MB of
metadata) and writes 1,507 Harbor task directories under
``~/.rllm/datasets/cybergym/``. Heavy ``data/`` archives (~236 GB) and
the ~10 TB of runner images are **not** pulled here — Docker ``ADD`` /
``FROM`` fetch them when a trial builds.

This is the official 1,507-task catalog (paper arXiv:2506.02548). It does
not ship research ARVO clones or draft toys.

    rllm dataset pull cybergym
    rllm eval cybergym --agent cybergym:claude-code

The 10-task official smoke set is ``cybergym-subset`` (Harbor
``--subset`` / CyberGym ``download_subset.py``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from rllm.integrations.cybergym.harbor_adapter import (
    SUBSET_TASK_IDS,
    CyberGymRecord,
    CyberGymToHarbor,
)
from rllm.integrations.cybergym.harbor_adapter.adapter import TEMPLATE_DIR
from rllm.integrations.cybergym.source import cybergym_task_to_row

logger = logging.getLogger(__name__)

HF_DATASET = "sunblaze-ucb/cybergym"
HF_TASKS_FILE = "tasks.json"
OFFICIAL_TASK_COUNT = 1507
DEFAULT_DIFFICULTY = "level1"

_DEFAULT_DESCRIPTION = "Official CyberGym Level 1 (1,507 Harbor tasks from sunblaze-ucb/cybergym). Thin Harbor runner. Not ExploitGym / ACE, not native :8666."


def _toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def load_official_records(
    *,
    tasks_json: str | Path | None = None,
    subset: bool = False,
    task_ids: Iterable[str] | None = None,
    exclude_task_ids: Iterable[str] | None = None,
    task_type: str | None = None,
    limit: int | None = None,
) -> list[CyberGymRecord]:
    """Load official CyberGym metadata (not the 236 GB ``data/`` tree)."""
    if tasks_json is not None:
        path = Path(tasks_json).expanduser()
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(HF_DATASET, HF_TASKS_FILE, repo_type="dataset"))
        raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a JSON list of CyberGym task records")

    records = [CyberGymRecord.from_dict(dict(row)) for row in raw]
    if not subset and task_ids is None and exclude_task_ids is None and task_type is None and limit is None and len(records) != OFFICIAL_TASK_COUNT:
        raise RuntimeError(f"Expected {OFFICIAL_TASK_COUNT} official CyberGym tasks in {path}, got {len(records)}. Refusing to register a truncated catalog.")

    if subset:
        keep = set(SUBSET_TASK_IDS)
        records = [r for r in records if r.task_id in keep]
        missing = keep.difference(r.task_id for r in records)
        if missing:
            raise RuntimeError(f"Official CyberGym subset IDs missing from metadata: {sorted(missing)}")

    if task_ids is not None:
        keep = set(task_ids)
        records = [r for r in records if r.task_id in keep]
        if not records:
            raise ValueError(f"No official CyberGym records for task_ids={sorted(keep)}")

    if exclude_task_ids is not None:
        drop = set(exclude_task_ids)
        records = [r for r in records if r.task_id not in drop]

    if task_type is not None:
        records = [r for r in records if r.task_type == task_type]

    if limit is not None:
        records = records[: int(limit)]

    return records


def _enrich_row(row: dict, record: CyberGymRecord) -> dict:
    row["source_task_id"] = record.task_id
    row["project_name"] = record.project_name
    row["arvo_project"] = record.project_name
    row["project_language"] = record.project_language
    row["task_type"] = record.task_type
    return row


def _row_for_record(task_dir: Path, record: CyberGymRecord) -> dict:
    row = cybergym_task_to_row(task_dir)
    if row is None:
        raise RuntimeError(f"Generated CyberGym task is not a Level 1 Harbor dir: {task_dir}")
    return _enrich_row(row, record)


def _write_dataset_toml(
    out: Path,
    *,
    name: str,
    split: str,
    description: str,
    default_agent: str,
) -> None:
    content = "\n".join(
        [
            "[dataset]",
            f'name = "{_toml_escape(name)}"',
            'type = "sandbox"',
            f'description = "{_toml_escape(description)}"',
            'default_sandbox = "docker"',
            f'default_agent = "{_toml_escape(default_agent)}"',
            f'split = "{_toml_escape(split)}"',
            "",
        ]
    )
    (out / "dataset.toml").write_text(content, encoding="utf-8")


def build_benchmark(
    name: str = "cybergym",
    split: str = "level1",
    out_dir: str | Path | None = None,
    catalog_entry: dict | None = None,
    *,
    subset: bool = False,
    difficulty: str = DEFAULT_DIFFICULTY,
    limit: int | None = None,
    task_ids: list[str] | None = None,
    exclude_task_ids: list[str] | None = None,
    task_type: str | None = None,
    tasks_json: str | Path | None = None,
    records: list[CyberGymRecord] | None = None,
    overwrite: bool = True,
    register: bool = True,
    default_agent: str = "cybergym:claude-code",
) -> Path:
    """Materialize official Harbor CyberGym Level 1 directories.

    Invoked from ``rllm dataset pull cybergym`` via the ``builder`` field
    in ``rllm/registry/datasets.json``.
    """
    catalog_entry = catalog_entry or {}
    split = catalog_entry.get("eval_split") or split
    default_agent = catalog_entry.get("default_agent") or default_agent
    kwargs = dict(catalog_entry.get("builder_kwargs") or {})
    subset = bool(kwargs.get("subset", subset))
    difficulty = str(kwargs.get("difficulty", difficulty))
    if limit is None and kwargs.get("limit") is not None:
        limit = int(kwargs["limit"])
    task_ids = task_ids if task_ids is not None else kwargs.get("task_ids")
    exclude_task_ids = exclude_task_ids if exclude_task_ids is not None else kwargs.get("exclude_task_ids")
    task_type = task_type if task_type is not None else kwargs.get("task_type")
    tasks_json = tasks_json if tasks_json is not None else kwargs.get("tasks_json")
    if kwargs.get("overwrite") is not None:
        overwrite = bool(kwargs["overwrite"])

    if out_dir is None:
        from rllm import paths

        out_dir = Path(paths.datasets_dir()) / name
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if records is None:
        logger.info("Loading official CyberGym metadata from %s/%s", HF_DATASET, HF_TASKS_FILE)
        records = load_official_records(
            tasks_json=tasks_json,
            subset=subset,
            task_ids=task_ids,
            exclude_task_ids=exclude_task_ids,
            task_type=task_type,
            limit=limit,
        )

    if not records:
        raise RuntimeError("No official CyberGym records to generate")

    converter = CyberGymToHarbor(
        output_dir=out,
        template_dir=TEMPLATE_DIR,
        difficulty=difficulty,
    )
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    for idx, record in enumerate(records, start=1):
        dest = out / record.task_dir_name
        try:
            if dest.exists() and not overwrite:
                logger.info("[%d] skip %s (exists)", idx, record.task_dir_name)
            else:
                dest = converter.generate_task(record, overwrite=overwrite)
                logger.info("[%d] OK   %s", idx, record.task_dir_name)
            rows.append(_row_for_record(dest, record))
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("[%d] FAIL %s: %s", idx, record.task_dir_name, msg)
            failures.append((record.task_dir_name, msg))

    if failures:
        detail = "; ".join(f"{name}: {reason}" for name, reason in failures[:8])
        extra = f" (+{len(failures) - 8} more)" if len(failures) > 8 else ""
        raise RuntimeError(f"Failed to generate {len(failures)} CyberGym tasks: {detail}{extra}")

    if not subset and limit is None and task_ids is None and exclude_task_ids is None and task_type is None:
        if len(rows) != OFFICIAL_TASK_COUNT:
            raise RuntimeError(f"Official CyberGym catalog must have {OFFICIAL_TASK_COUNT} tasks, generated {len(rows)}")

    description = catalog_entry.get("description") or _DEFAULT_DESCRIPTION
    _write_dataset_toml(
        out,
        name=name,
        split=split,
        description=description,
        default_agent=default_agent,
    )

    logger.info("Wrote %d official CyberGym Harbor tasks to %s", len(rows), out)

    if register:
        from rllm.data import DatasetRegistry

        DatasetRegistry.register_dataset(
            name=name,
            data=rows,
            split=split,
            source=catalog_entry.get("source") or HF_DATASET,
            description=description,
            category=catalog_entry.get("category", "agentic"),
        )

    return out


def generate_split(split: str, catalog_entry: dict) -> list[dict]:
    """Generate official Harbor dirs and return catalog rows.

    Kept for callers that want rows without going through DatasetRegistry.
    ``catalog_entry['out_dir']`` / ``builder_kwargs`` control generation.
    """
    out = catalog_entry.get("out_dir") or (catalog_entry.get("builder_kwargs") or {}).get("out_dir")
    if not out:
        from rllm import paths

        out = Path(paths.datasets_dir()) / "cybergym"
    root = build_benchmark(
        name=str(catalog_entry.get("name") or "cybergym"),
        split=split,
        out_dir=out,
        catalog_entry=catalog_entry,
        register=False,
    )
    from rllm.integrations.cybergym.source import load_cybergym_rows

    rows = load_cybergym_rows(root)
    if not rows:
        raise RuntimeError(f"No official CyberGym Level 1 task directories under {root}")
    return rows


def main() -> None:
    """CLI: ``python -m rllm.data.cybergym_builder --out-dir <dir> [--subset]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate official Harbor CyberGym Level 1 task directories.")
    parser.add_argument("--out-dir", required=True, help="Output directory for Harbor task dirs.")
    parser.add_argument("--name", default="cybergym")
    parser.add_argument("--split", default="level1")
    parser.add_argument("--subset", action="store_true", help="Official 10-task smoke subset.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--tasks-json", default=None, help="Local tasks.json (skip HuggingFace).")
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level="INFO")
    build_benchmark(
        name=args.name,
        split=args.split,
        out_dir=args.out_dir,
        subset=args.subset,
        limit=args.limit,
        task_ids=args.task_ids,
        tasks_json=args.tasks_json,
        overwrite=not args.no_overwrite,
        register=False,
    )


if __name__ == "__main__":
    main()

"""Load Harbor CyberGym task directories as rLLM dataset rows."""

from __future__ import annotations

import logging
from pathlib import Path

import tomllib

from rllm.integrations.harbor.cybergym.constants import DEFAULT_SCORING_MODE, LEVEL1
from rllm.integrations.harbor.cybergym.detect import discover_cybergym_task_dirs, is_cybergym_task_dir

__all__ = [
    "cybergym_task_to_row",
    "load_cybergym_rows",
    "source_task_id_from_dir_name",
]

logger = logging.getLogger(__name__)


def source_task_id_from_dir_name(name: str) -> str:
    """``cybergym_arvo_1065`` → ``arvo:1065``; ``cybergym_oss-fuzz_42535201`` → ``oss-fuzz:…``."""
    if not name.startswith("cybergym_"):
        return ""
    rest = name[len("cybergym_") :]
    if rest.startswith("oss-fuzz_"):
        return "oss-fuzz:" + rest[len("oss-fuzz_") :]
    if rest.startswith("arvo_"):
        return "arvo:" + rest[len("arvo_") :]
    return ""


def cybergym_task_to_row(task_dir: str | Path) -> dict | None:
    """Convert one Harbor CyberGym task directory into a dataset row."""
    path = Path(task_dir).resolve()
    if not is_cybergym_task_dir(path):
        logger.warning("Skipping non-CyberGym Level 1 task dir: %s", path)
        return None

    instruction = (path / "instruction.md").read_text(encoding="utf-8")
    raw: dict = {}
    try:
        raw = tomllib.loads((path / "task.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Could not parse %s/task.toml: %s", path, exc)

    task_section = raw.get("task") or {}
    meta_section = raw.get("metadata") or {}
    name = task_section.get("name") or path.name
    source_task_id = meta_section.get("source_task_id") or source_task_id_from_dir_name(path.name)
    return {
        "task_id": name,
        "task_path": str(path),
        "instruction": instruction,
        "question": instruction,
        "ground_truth": None,
        "data_source": f"cybergym:{name}",
        "cybergym_level": LEVEL1,
        "cybergym_scoring": DEFAULT_SCORING_MODE,
        "harbor_task_name": name,
        "harbor_category": meta_section.get("category") or "cybersecurity",
        "source_task_id": source_task_id,
        "arvo_project": meta_section.get("arvo_project") or meta_section.get("project_name") or "",
        "fuzz_target": meta_section.get("fuzz_target") or "",
    }


def load_cybergym_rows(root: str | Path) -> list[dict]:
    """Load every Level 1 CyberGym task under *root* as Harbor-style rows."""
    rows: list[dict] = []
    for task_dir in discover_cybergym_task_dirs(root):
        row = cybergym_task_to_row(task_dir)
        if row is not None:
            rows.append(row)
    logger.info("Loaded %d CyberGym Level 1 tasks from %s", len(rows), root)
    return rows

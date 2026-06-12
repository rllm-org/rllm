"""Builder for the SkillsBench sandbox benchmark.

SkillsBench (``benchflow/skillsbench``) ships its tasks as a single
Parquet file where each row inlines a complete Harbor-format task tree
(``task.toml``, ``instruction.md``, ``environment/Dockerfile``,
``tests/test.sh`` + ``tests/test_outputs.py``, ``solution/solve.sh``,
plus an arbitrary ``files`` list and a ``skills`` list).

This module expands each row into rLLM's sandbox (task-per-directory)
benchmark layout so ``rllm eval`` runs each task through the standard
``SandboxedAgentFlow`` + ``ShellScriptEvaluator`` path. No Harbor runtime
involvement — the tasks just happen to use the Harbor task format on
disk, which the rLLM ``BenchmarkLoader`` already understands.

Skills injection: SkillsBench task authors place skill scripts under
``environment/skills/<name>/scripts/`` but the row's Dockerfile does
*not* COPY them — they assume the harness (Harbor's, or ours) mounts
them at ``/root/.claude/skills/`` at runtime. To keep the
rllm-native execution path image-driven (no host-side mounts), the
builder writes each skill's ``SKILL.md`` into the Docker build context
alongside the existing scripts and appends a single ``COPY skills
/root/.claude/skills/`` line to each task's Dockerfile when skills are
present and ``include_skills`` is True.

Invoked from:
- ``rllm dataset pull skillsbench`` (via the ``builder`` field in
  ``rllm/registry/datasets.json`` → :func:`rllm.cli._pull.pull_dataset`).

On-disk output (``<out_dir>/``):

    skillsbench/
    ├── dataset.toml                                   # type="sandbox"
    ├── <task_id>/
    │   ├── task.toml
    │   ├── instruction.md
    │   ├── environment/Dockerfile                     # +COPY skills line
    │   ├── environment/skills/<name>/SKILL.md         # skill body
    │   ├── environment/skills/<name>/scripts/...      # from row.files
    │   ├── tests/test.sh
    │   ├── tests/test_outputs.py                      # canonical filename
    │   ├── solution/solve.sh
    │   └── ... other files from row.files ...
    └── ...

Grading: each task's own ``tests/test.sh`` is the verifier. rLLM's
``ShellScriptEvaluator`` runs it inside the sandbox and reads the
reward from ``/tmp/rllm/reward.json`` (or the Harbor-compatible
fallbacks).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ID = "benchflow/skillsbench"
PARQUET_FILENAME = "skillsbench-tasks.parquet"

# Columns whose contents map to canonical paths inside each task dir.
# Order matters only for logging.
_INLINED_FILES: tuple[tuple[str, str], ...] = (
    ("task_toml", "task.toml"),
    ("instruction", "instruction.md"),
    ("dockerfile", "environment/Dockerfile"),
    ("solve_sh", "solution/solve.sh"),
    ("test_sh", "tests/test.sh"),
)

# Files that need the executable bit set after writing.
_EXECUTABLE_PATHS: frozenset[str] = frozenset(
    {
        "solution/solve.sh",
        "tests/test.sh",
    }
)


def _toml_escape(s: str) -> str:
    """Escape a string for a TOML triple-quoted basic string."""
    return s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def _coerce_text(value: Any) -> str | None:
    """Decode a parquet cell to a Python string. ``None`` if missing/empty."""
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_bytes(value: Any) -> bytes | None:
    """Decode a parquet cell to bytes. ``None`` if missing."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


def _write_inlined(task_dir: Path, row: dict, column: str, rel_path: str) -> bool:
    """Write a single inlined file. Returns True if written."""
    text = _coerce_text(row.get(column))
    if text is None:
        return False
    target = task_dir / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    if rel_path in _EXECUTABLE_PATHS:
        target.chmod(0o755)
    return True


def _write_test_outputs(task_dir: Path, row: dict) -> None:
    """Write the test fixture script alongside tests/test.sh.

    SkillsBench stores its name as ``test_outputs`` but the actual filename
    on disk in the upstream repos is ``test_outputs.py``. We write it under
    that canonical name; ``tests/test.sh`` references it by relative path.
    """
    text = _coerce_text(row.get("test_outputs"))
    if text is None:
        return
    target = task_dir / "tests" / "test_outputs.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _write_files_list(task_dir: Path, row: dict) -> int:
    """Materialize the row's ``files`` column.

    Each entry is a struct with keys ``path``, ``content``, ``is_text``,
    ``size_bytes``, ``sha256``. Returns the number of files written.
    """
    files = row.get("files") or []
    written = 0
    for entry in files:
        path = entry.get("path") if isinstance(entry, dict) else None
        if not path:
            continue
        rel = path.lstrip("/")
        target = task_dir / rel
        if not _is_within(task_dir, target):
            logger.warning("Skipping unsafe path %s in %s", path, task_dir.name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        is_text = bool(entry.get("is_text"))
        if is_text:
            text = _coerce_text(entry.get("content"))
            if text is None:
                continue
            target.write_text(text, encoding="utf-8")
        else:
            data = _coerce_bytes(entry.get("content"))
            if data is None:
                continue
            target.write_bytes(data)
        # Preserve executable bit for shell-script-looking siblings.
        if rel.endswith(".sh"):
            target.chmod(0o755)
        written += 1
    return written


def _write_skills(task_dir: Path, row: dict) -> int:
    """Materialize the row's ``skills`` column inside the Docker build context.

    Each entry is a struct with ``name``, ``description``, ``skill_md``. The
    body is written to ``environment/skills/<safe_name>/SKILL.md`` so it sits
    alongside any per-skill ``scripts/`` files already placed by
    :func:`_write_files_list`, and so a single ``COPY skills
    /root/.claude/skills/`` line in the Dockerfile pulls both into the image
    at the path SkillsBench solve.sh scripts expect.

    Returns the number of skill bodies written.
    """
    skills = row.get("skills") or []
    if not skills:
        return 0
    skills_root = task_dir / "environment" / "skills"
    written = 0
    for entry in skills:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        body = _coerce_text(entry.get("skill_md"))
        if not name or body is None:
            continue
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))
        skill_dir = skills_root / safe_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
        written += 1
    return written


# Marker so we never double-patch a Dockerfile that's been processed already
# (e.g. when rerunning the builder without --clean).
_SKILLS_COPY_MARKER = "# rllm-skillsbench: skills injection"


def _patch_dockerfile_for_skills(task_dir: Path) -> bool:
    """Append a ``COPY skills /root/.claude/skills/`` to the task's Dockerfile.

    No-op if the Dockerfile is missing, if there's no ``environment/skills/``
    directory to copy from, or if the marker is already present. Returns
    True when a patch was applied.
    """
    dockerfile = task_dir / "environment" / "Dockerfile"
    skills_dir = task_dir / "environment" / "skills"
    if not dockerfile.exists() or not skills_dir.is_dir():
        return False
    text = dockerfile.read_text()
    if _SKILLS_COPY_MARKER in text:
        return False
    suffix = f"\n\n{_SKILLS_COPY_MARKER}\nCOPY skills /root/.claude/skills/\n"
    if not text.endswith("\n"):
        suffix = "\n" + suffix
    dockerfile.write_text(text + suffix)
    return True


def _is_within(root: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


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
            f'name = "{name}"',
            'type = "sandbox"',
            f'description = "{_toml_escape(description)}"',
            'default_sandbox = "docker"',
            f'default_agent = "{default_agent}"',
            f'split = "{split}"',
            "",
        ]
    )
    (out / "dataset.toml").write_text(content, encoding="utf-8")


def _strip_skills_for_no_skills_variant(row: dict) -> dict:
    """Return a copy of *row* with all skill traces removed.

    Drops the ``skills`` column outright and filters any ``files`` entries
    whose ``path`` lives under ``environment/skills/``. Used by the
    ``skillsbench-no-skills`` catalog variant to make the agent solve the
    task with no skill scaffolding available in the image.
    """
    files = row.get("files") or []
    filtered = []
    for entry in files:
        if not isinstance(entry, dict):
            filtered.append(entry)
            continue
        path = (entry.get("path") or "").lstrip("/")
        if path.startswith("environment/skills/") or path.startswith("skills/"):
            continue
        filtered.append(entry)
    return {**row, "skills": [], "files": filtered}


def _materialize_task(task_dir: Path, row: dict) -> dict:
    """Expand a single SkillsBench row into a Harbor-format task tree.

    Returns a small stats dict useful for logging.
    """
    task_dir.mkdir(parents=True, exist_ok=True)
    stats = {"inlined": 0, "files": 0, "skills": 0, "patched_dockerfile": False}
    for column, rel_path in _INLINED_FILES:
        if _write_inlined(task_dir, row, column, rel_path):
            stats["inlined"] += 1
    _write_test_outputs(task_dir, row)
    stats["files"] = _write_files_list(task_dir, row)
    stats["skills"] = _write_skills(task_dir, row)
    # The Dockerfile-patch lives outside _write_skills so it also fires when
    # the row carries skill scripts via files[] but no skill bodies, or vice
    # versa — as long as there's anything under environment/skills/, expose
    # it to the agent at the conventional /root/.claude/skills/ path.
    stats["patched_dockerfile"] = _patch_dockerfile_for_skills(task_dir)
    return stats


def _read_parquet_rows(parquet_path: Path) -> list[dict]:
    """Load the SkillsBench parquet into a list of plain dict rows.

    Uses ``pyarrow`` so the struct/list-of-struct columns (``files``,
    ``skills``) come back as nested Python objects.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    return table.to_pylist()


def build_benchmark(
    *,
    name: str = "skillsbench",
    split: str = "train",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    default_agent: str = "claude-code",
    include_skills: bool = True,
    clean: bool = False,
    register: bool = True,
) -> Path:
    """Materialize SkillsBench into a sandbox benchmark directory.

    Args:
        name: Dataset/registry name (also the dataset.toml ``name``).
        split: HF split to build (SkillsBench ships a single ``train`` split).
        out_dir: Output benchmark directory.
        catalog_entry: Optional catalog entry (datasets.json); ``description``,
            ``default_agent``, ``eval_split`` are read from it when present.
        task_ids: Optional list of task IDs to keep. ``None`` means all.
        limit: Keep only the first N tasks (after the task_ids filter).
        default_agent: ``default_agent`` written into dataset.toml.
        include_skills: If False, omit the per-task ``skills/`` tree so the
            agent has to solve tasks without skill-augmented prompts. Lets the
            same builder back two catalog entries: ``skillsbench`` and
            ``skillsbench-no-skills``.
        clean: Remove ``out_dir`` before building.
        register: Also register rows in ``DatasetRegistry`` so
            ``rllm dataset list/info/inspect`` show the dataset as pulled.

    Returns:
        Path to the built benchmark directory.
    """
    from huggingface_hub import hf_hub_download

    if catalog_entry:
        split = catalog_entry.get("eval_split") or split
        default_agent = catalog_entry.get("default_agent") or default_agent

    out = Path(out_dir).expanduser()
    if clean and out.exists():
        logger.info("[skillsbench] removing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("[skillsbench] downloading %s/%s (cached) ...", REPO_ID, PARQUET_FILENAME)
    parquet_path = Path(hf_hub_download(REPO_ID, PARQUET_FILENAME, repo_type="dataset"))

    logger.info("[skillsbench] loading rows from %s ...", parquet_path.name)
    rows = _read_parquet_rows(parquet_path)

    if task_ids is not None:
        keep = set(task_ids)
        rows = [r for r in rows if r.get("task_id") in keep]
    if limit is not None:
        rows = rows[:limit]

    logger.info("[skillsbench] selected %d tasks (task_ids=%s, limit=%s)", len(rows), task_ids and len(task_ids), limit)

    total_files = 0
    total_skills = 0
    patched = 0
    written_tasks = 0
    for row in rows:
        task_id = row.get("task_id")
        if not task_id:
            logger.warning("[skillsbench] row missing task_id, skipping")
            continue
        task_dir = out / str(task_id)
        # Wipe per-task dir so reruns don't leave stale fixtures behind.
        if task_dir.exists():
            shutil.rmtree(task_dir)
        if not include_skills:
            # Strip skills before materializing so _write_skills sees nothing
            # AND so files[] entries placing tooling under environment/skills/
            # are dropped — otherwise the Dockerfile patch would still fire
            # off the leftover scripts dir.
            row = _strip_skills_for_no_skills_variant(row)
        stats = _materialize_task(task_dir, row)
        total_files += stats["files"]
        total_skills += stats["skills"]
        patched += int(stats["patched_dockerfile"])
        written_tasks += 1

    description = (catalog_entry or {}).get("description") or "SkillsBench: 91 expert-curated agentic tasks measuring AI agents' use of skills."
    _write_dataset_toml(
        out,
        name=name,
        split=split,
        description=description,
        default_agent=default_agent,
    )

    logger.info(
        "[skillsbench] wrote %d task dirs to %s (%d extra files, %d skill docs, %d Dockerfiles patched, include_skills=%s)",
        written_tasks,
        out,
        total_files,
        total_skills,
        patched,
        include_skills,
    )

    if register:
        try:
            from rllm.data import DatasetRegistry

            reg_rows = []
            for r in rows:
                if not r.get("task_id"):
                    continue
                reg_rows.append(
                    {
                        "task_id": r.get("task_id"),
                        "category": r.get("category", ""),
                        "difficulty": r.get("difficulty", ""),
                        "tags": list(r.get("tags") or []),
                        "allow_internet": bool(r.get("allow_internet", False)),
                    }
                )
            DatasetRegistry.register_dataset(
                name=name,
                data=reg_rows,
                split=split,
                source=REPO_ID,
                description=description,
                category=(catalog_entry or {}).get("category", "agentic"),
            )
        except Exception:
            logger.warning(
                "[skillsbench] could not register rows in DatasetRegistry (non-fatal)",
                exc_info=True,
            )

    return out


def main() -> None:
    """CLI for direct invocation: ``python -m rllm.data.skillsbench_builder``."""
    import argparse

    parser = argparse.ArgumentParser(description="Materialize SkillsBench into an rLLM sandbox benchmark directory.")
    parser.add_argument("--out-dir", required=True, help="Output benchmark directory.")
    parser.add_argument("--name", default="skillsbench")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--default-agent", default="claude-code")
    parser.add_argument("--no-skills", action="store_true", help="Omit the per-task skills/ tree.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=os.environ.get("RLLM_LOG_LEVEL", "INFO"))
    build_benchmark(
        name=args.name,
        split=args.split,
        out_dir=args.out_dir,
        task_ids=args.task_ids,
        limit=args.limit,
        default_agent=args.default_agent,
        include_skills=not args.no_skills,
        clean=args.clean,
        register=False,
    )


if __name__ == "__main__":
    main()

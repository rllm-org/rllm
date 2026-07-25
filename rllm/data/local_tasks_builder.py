"""Builder for datasets made from a local directory of Harbor task dirs.

Unlike the HF-sourced builders (``tmax_builder``, ``r2egym_builder``, ...) this
builder has nothing to download: the tasks already exist on disk in Harbor's
directory-per-task layout (``task.toml`` + ``instruction.md`` +
``environment/`` + ``tests/test.sh``), e.g. ``~/mono/tb_v2``. It selects a
subset of those task dirs, registers them as rows in the
:class:`~rllm.data.dataset.DatasetRegistry` (the parquet ``train.py`` loads),
and symlinks the selected dirs under ``<out_dir>/`` with a ``dataset.toml`` so
the ``rllm eval`` materialized-benchmark redirect also resolves the name.

Rows keep ``task_path`` pointing at the REAL task dirs under ``tasks_root`` —
tasks are never copied. Keep ``tasks_root`` in place (and present at the same
path on whatever machine training runs on).

Task selection (``builder_kwargs`` in the catalog entry, or direct kwargs):

* ``task_ids`` — explicit list of task dir names under ``tasks_root``.
* ``task_list_file`` — path to a file with one entry per line: a plain task
  name, an absolute path, or a ``{"Path": "..."}`` JSON object.
* neither — every valid task dir under ``tasks_root`` (optionally ``limit``-ed).

Back multiple dataset versions off this one builder with different catalog
entries, e.g.::

    "tb-v2-subset8":  {"builder": "rllm.data.local_tasks_builder:build_benchmark",
                       "builder_kwargs": {"tasks_root": "~/mono/tb_v2",
                                           "task_ids": ["bottleneck-path-oracle", ...]}},
    "tb-v2-full":     {"builder": "rllm.data.local_tasks_builder:build_benchmark",
                       "builder_kwargs": {"tasks_root": "~/mono/tb_v2"}}

Invoked from ``rllm dataset pull <name>`` via the ``builder`` field in
``rllm/registry/datasets.json`` → :func:`rllm.cli._pull.pull_dataset`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_task_list_file(path: Path) -> list[str]:
    """One entry per line: plain task name, absolute path, or {"Path": ...} JSON."""
    entries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            line = json.loads(line)["Path"]
        entries.append(line)
    return entries


def _select_task_dirs(
    tasks_root: Path,
    task_ids: list[str] | None,
    task_list_file: str | None,
    limit: int | None,
) -> list[Path]:
    """Resolve the selection kwargs to a sorted list of task dirs."""
    if task_ids is not None and task_list_file is not None:
        raise ValueError("Pass task_ids OR task_list_file, not both")

    entries: list[str] | None = task_ids
    if task_list_file is not None:
        entries = _parse_task_list_file(Path(task_list_file).expanduser())

    if entries is None:
        dirs = sorted(d for d in tasks_root.iterdir() if d.is_dir() and (d / "task.toml").exists())
        return dirs[:limit] if limit is not None else dirs

    dirs, missing = [], []
    for e in entries:
        p = Path(e).expanduser()
        candidate = p if p.is_absolute() else tasks_root / Path(e).name
        if not (candidate / "task.toml").exists():
            candidate = tasks_root / Path(e).name
        if (candidate / "task.toml").exists():
            dirs.append(candidate)
        else:
            missing.append(e)
    if missing:
        raise FileNotFoundError(f"{len(missing)} task(s) not found under {tasks_root}: {missing[:10]}")
    return dirs[:limit] if limit is not None else dirs


def _write_dataset_toml(out: Path, *, name: str, split: str, description: str, default_agent: str) -> None:
    content = "\n".join(
        [
            "[dataset]",
            f'name = "{name}"',
            'type = "sandbox"',
            f'description = "{description}"',
            'default_sandbox = "docker"',
            f'default_agent = "{default_agent}"',
            f'split = "{split}"',
            "",
            "[verifier]",
            'script = "tests/test.sh"',
            "",
        ]
    )
    (out / "dataset.toml").write_text(content, encoding="utf-8")


def build_benchmark(
    *,
    name: str,
    split: str = "train",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    tasks_root: str = "~/mono/tb_v2",
    task_ids: list[str] | None = None,
    task_list_file: str | None = None,
    limit: int | None = None,
    default_agent: str = "terminus-2",
    register: bool = True,
) -> Path:
    """Register a subset of an on-disk Harbor task tree as a named dataset."""
    from rllm.integrations.harbor.dataset_loader import harbor_task_to_row

    if catalog_entry:
        default_agent = catalog_entry.get("default_agent") or default_agent

    root = Path(tasks_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"tasks_root does not exist: {root}")

    task_dirs = _select_task_dirs(root, task_ids, task_list_file, limit)
    if not task_dirs:
        raise RuntimeError(f"No Harbor task directories (task.toml) selected under {root}")

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped = 0
    for d in task_dirs:
        row = harbor_task_to_row(d)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        link = out / d.name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(d)

    if not rows:
        raise RuntimeError(f"All {len(task_dirs)} selected task dirs under {root} were invalid/skipped")
    if skipped:
        logger.warning("[local-tasks] skipped %d invalid/multi-step task(s)", skipped)

    description = (catalog_entry or {}).get("description") or f"Local Harbor tasks ({len(rows)} tasks from {root})"
    _write_dataset_toml(out, name=name, split=split, description=description, default_agent=default_agent)
    logger.info("[local-tasks] linked %d task dirs into %s", len(rows), out)

    if register:
        from rllm.data import DatasetRegistry

        DatasetRegistry.register_dataset(
            name=name,
            data=rows,
            split=split,
            source=f"local:{root}",
            description=description,
            category=(catalog_entry or {}).get("category", "agentic"),
        )
        logger.info("[local-tasks] registered %s/%s (%d tasks)", name, split, len(rows))

    return out

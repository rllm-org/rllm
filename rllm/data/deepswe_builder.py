"""Builder for the DeepSWE sandbox benchmark.

DeepSWE (``datacurve-ai/deep-swe``) ships its 113 long-horizon software
engineering tasks as Harbor-format task directories under ``tasks/`` in a
plain GitHub repo (no HuggingFace mirror). Each task already follows
rLLM's ``type="sandbox"`` task-per-directory shape:

    deep-swe/tasks/<task_id>/
    ├── task.toml
    ├── instruction.md
    ├── environment/Dockerfile
    ├── tests/test.sh          # the verifier
    ├── tests/test.patch       # applied by the verifier (per-task fixture)
    └── solution/              # reference solution (withheld from agents)

Building the benchmark is therefore: shallow-clone the repo, copy each
selected task dir into the output, and write a ``dataset.toml``.

Single source of truth for materializing the dataset, used by
``rllm dataset pull deepswe`` (via the ``builder`` field in
``rllm/registry/datasets.json`` → :func:`rllm.cli._pull.pull_dataset`).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/datacurve-ai/deep-swe.git"
TASKS_SUBDIR = "tasks"


def _git_clone_tasks() -> Path:
    """Shallow-clone the deep-swe repo into a temp dir and return its tasks/ path.

    Uses ``--depth 1`` so we only fetch the latest snapshot — the 113 tasks
    are large enough (each ships a Dockerfile + verifier fixtures) that
    pulling history would be wasteful, and reproducibility is anchored at
    the per-task ``base_commit`` recorded in each ``task.toml`` rather than
    the deep-swe repo HEAD.
    """
    tmp = Path(tempfile.mkdtemp(prefix="deepswe-"))
    logger.info("[deepswe] cloning %s into %s ...", REPO_URL, tmp)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(tmp)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        out = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
        raise RuntimeError(f"git clone of {REPO_URL} failed:\n{out}") from e
    tasks_dir = tmp / TASKS_SUBDIR
    if not tasks_dir.is_dir():
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"Expected {TASKS_SUBDIR}/ in cloned repo at {tmp}")
    return tasks_dir


def _discover_task_dirs(tasks_dir: Path) -> list[Path]:
    """Return every immediate subdirectory of ``tasks/`` that has a task.toml."""
    return sorted(d for d in tasks_dir.iterdir() if d.is_dir() and (d / "task.toml").exists())


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
    name: str = "deepswe",
    split: str = "test",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    default_agent: str = "mini-swe-agent",
    clean: bool = False,
    register: bool = True,
) -> Path:
    """Materialize datacurve-ai/deep-swe into a sandbox benchmark directory.

    Args:
        name: Dataset/registry name (also the dataset.toml ``name``).
        split: Split label written into dataset.toml and the registry.
        out_dir: Output benchmark directory.
        catalog_entry: Optional catalog entry (datasets.json); ``description``
            and ``default_agent`` are read from it when present.
        task_ids: Build only these task ids. Default: all 113 tasks.
        limit: Keep only the first N tasks (after the ``task_ids`` filter).
        default_agent: ``default_agent`` written into dataset.toml.
        clean: Remove ``out_dir`` before building.
        register: Also register ``task_path`` rows in ``DatasetRegistry`` so
            the name-based eval/train flows and ``rllm dataset list`` work.

    Returns:
        Path to the built benchmark directory.
    """
    if catalog_entry:
        default_agent = catalog_entry.get("default_agent") or default_agent

    out = Path(out_dir).expanduser()
    if clean and out.exists():
        logger.info("[deepswe] removing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    clone_root = _git_clone_tasks()
    try:
        all_task_dirs = _discover_task_dirs(clone_root)
        if not all_task_dirs:
            raise RuntimeError(f"No task directories with task.toml under {clone_root}")

        if task_ids is not None:
            wanted = set(task_ids)
            selected = [d for d in all_task_dirs if d.name in wanted]
            missing = wanted - {d.name for d in selected}
            if missing:
                logger.warning("[deepswe] task_ids not found in repo: %s", sorted(missing))
        else:
            selected = all_task_dirs

        if limit is not None:
            selected = selected[:limit]

        for d in selected:
            dst = out / d.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(d, dst)

        description = (catalog_entry or {}).get("description") or "DeepSWE: 113 long-horizon SWE tasks across 91 repos in 5 languages (Harbor format; in-sandbox verifier)."
        _write_dataset_toml(
            out,
            name=name,
            split=split,
            description=description,
            default_agent=default_agent,
        )
        logger.info("[deepswe] wrote %d task dirs to %s", len(selected), out)

        if register:
            try:
                from rllm.data import DatasetRegistry

                reg_rows = []
                for d in selected:
                    task_dst = out / d.name
                    instruction_path = task_dst / "instruction.md"
                    instruction = instruction_path.read_text(encoding="utf-8") if instruction_path.exists() else ""
                    reg_rows.append(
                        {
                            "id": d.name,
                            "instruction": instruction,
                            "task_path": str(task_dst),
                        }
                    )
                DatasetRegistry.register_dataset(
                    name=name,
                    data=reg_rows,
                    split=split,
                    source=REPO_URL,
                    description=description,
                    category=(catalog_entry or {}).get("category", "agentic"),
                )
            except Exception:
                logger.warning("[deepswe] could not register rows in DatasetRegistry (non-fatal)", exc_info=True)

        return out
    finally:
        shutil.rmtree(clone_root.parent, ignore_errors=True)


def main() -> None:
    """CLI for direct invocation: ``python -m rllm.data.deepswe_builder``."""
    import argparse

    parser = argparse.ArgumentParser(description="Materialize DeepSWE into an rLLM sandbox benchmark directory.")
    parser.add_argument("--out-dir", required=True, help="Output benchmark directory.")
    parser.add_argument("--name", default="deepswe")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--default-agent", default="mini-swe-agent")
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
        clean=args.clean,
        register=False,
    )


if __name__ == "__main__":
    main()

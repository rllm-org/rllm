"""Builder for the Claw-Eval sandbox benchmark.

Claw-Eval (``claw-eval/Claw-Eval``) is a personal-assistant agent benchmark
whose tasks are *workspaces*: a natural-language ``query`` plus fixture files.
This module materializes a split into rLLM's ``type="sandbox"`` task-per-
directory layout so ``rllm eval`` can run each task in a sandbox.

It is the single source of truth for building the dataset, used by:
- ``rllm dataset pull claw_eval`` (via the ``builder`` field in
  ``rllm/registry/datasets.json`` → :func:`rllm.cli._pull.pull_dataset`), and
- ``scripts/data/claw_eval_dataset.py`` (a thin CLI wrapper with extra knobs).

On-disk output (``<out_dir>/``):

    claw_eval/
    ├── dataset.toml                     # [dataset] type="sandbox", default_agent="zeroclaw"
    ├── <task_id>/
    │   ├── task.toml                    # top-level query/rubric + [environment]/[verifier]
    │   ├── instruction.md               # the row's query
    │   └── environment/files/fixtures/  # this task's fixtures (uploaded to /workspace)
    └── ...

Grading: rows ship no rubric, so each task's ``[verifier]`` points at
``claw_eval_reward_fn`` (an LLM-judge over the agent transcript).
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ID = "claw-eval/Claw-Eval"
VERIFIER_NAME = "claw_eval_reward_fn"


def _toml_escape(s: str) -> str:
    """Escape a string for a TOML triple-quoted basic string."""
    return s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def _write_task(task_dir: Path, *, task_id: str, query: str, category: str, language: str, judge_model: str | None) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "instruction.md").write_text(query, encoding="utf-8")

    # Top-level scalars land directly in Task.metadata (the loader copies the
    # whole task.toml into metadata), so the judge can read `query`/`rubric`.
    # Scalars MUST precede the first [section] in TOML.
    lines = [
        f'query = """{_toml_escape(query)}"""',
        f'rubric = """{_toml_escape(query)}"""',
        f'task_id = "{task_id}"',
        f'category = "{category}"',
        f'language = "{language}"',
    ]
    if judge_model:
        lines.append(f'judge_model = "{judge_model}"')
    lines += [
        "",
        "[task]",
        f'name = "{task_id}"',
        "",
        "[environment]",
        'workdir = "/workspace"',
        "",
        "[verifier]",
        f'name = "{VERIFIER_NAME}"',
        "",
    ]
    (task_dir / "task.toml").write_text("\n".join(lines), encoding="utf-8")


def _write_dataset_toml(out: Path, *, name: str, split: str, description: str, default_agent: str) -> None:
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


def build_benchmark(
    *,
    name: str = "claw_eval",
    split: str = "general",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    limit: int | None = None,
    lang: str = "all",
    default_agent: str = "zeroclaw",
    judge_model: str | None = None,
    clean: bool = False,
    register: bool = True,
) -> Path:
    """Materialize a Claw-Eval split into a sandbox benchmark directory.

    Args:
        name: Dataset/registry name (also the dataset.toml ``name``).
        split: HF split to build (``general`` | ``multimodal`` | ``multi_turn``).
        out_dir: Output benchmark directory.
        catalog_entry: Optional catalog entry (datasets.json); ``description``/
            ``default_agent``/``eval_split`` are read from it when present.
        limit: Keep only the first N tasks (after the language filter).
        lang: ``all`` | ``en`` | ``zh`` language filter.
        default_agent: ``default_agent`` written into dataset.toml.
        judge_model: Optional judge model stamped into each task's metadata.
        clean: Remove ``out_dir`` before building.
        register: Also register the rows in ``DatasetRegistry`` so
            ``rllm dataset list/info/inspect`` show the dataset as pulled.

    Returns:
        Path to the built benchmark directory.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    if catalog_entry:
        split = catalog_entry.get("eval_split") or split
        default_agent = catalog_entry.get("default_agent") or default_agent

    out = Path(out_dir).expanduser()
    if clean and out.exists():
        logger.info("[claw-eval] removing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("[claw-eval] loading %s split=%s ...", REPO_ID, split)
    ds = load_dataset(REPO_ID, split=split)

    rows = list(ds)
    if lang != "all":
        rows = [r for r in rows if r["language"] == lang]
    if limit is not None:
        rows = rows[:limit]
    selected_ids = {r["task_id"] for r in rows}
    logger.info("[claw-eval] selected %d tasks (lang=%s, limit=%s)", len(rows), lang, limit)

    # Extract only the fixtures for the selected tasks from the shared archive.
    logger.info("[claw-eval] downloading fixtures archive (cached) ...")
    fixtures_path = hf_hub_download(REPO_ID, "data/fixtures.tar.gz", repo_type="dataset")

    dest_files = {r["task_id"]: out / r["task_id"] / "environment" / "files" for r in rows}
    extracted = dict.fromkeys(selected_ids, 0)
    logger.info("[claw-eval] extracting fixtures ...")
    with tarfile.open(fixtures_path) as tar:
        for m in tar:
            if not m.isfile():
                continue
            top = m.name.split("/", 1)[0]
            if top not in selected_ids:
                continue
            rel = m.name[len(top) + 1 :]  # strip "<task_id>/" -> "fixtures/..."
            if not rel:
                continue
            target = dest_files[top] / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src is None:
                continue
            with open(target, "wb") as fh:
                shutil.copyfileobj(src, fh)
            extracted[top] += 1

    for r in rows:
        _write_task(
            out / r["task_id"],
            task_id=r["task_id"],
            query=r["query"],
            category=r["category"],
            language=r["language"],
            judge_model=judge_model,
        )

    description = (catalog_entry or {}).get("description") or f"Claw-Eval {split} split: personal-assistant agent tasks (LLM-judge graded)."
    _write_dataset_toml(out, name=name, split=split, description=description, default_agent=default_agent)

    n_with_fx = sum(1 for tid in selected_ids if extracted[tid] > 0)
    logger.info("[claw-eval] wrote %d task dirs to %s (%d with fixtures, %d fixture-less)", len(rows), out, n_with_fx, len(selected_ids) - n_with_fx)

    # Register the rows so `rllm dataset list/info/inspect` show parity with
    # other catalog datasets. Eval still uses the sandbox dir (via the
    # materialised-benchmark redirect in rllm.cli.eval), not these rows.
    if register:
        try:
            from rllm.data import DatasetRegistry

            reg_rows = [{k: r[k] for k in ("task_id", "query", "fixture", "language", "category")} for r in rows]
            DatasetRegistry.register_dataset(
                name=name,
                data=reg_rows,
                split=split,
                source=REPO_ID,
                description=description,
                category=(catalog_entry or {}).get("category", "agentic"),
            )
        except Exception:  # registry parity is best-effort; the benchmark dir is what eval uses
            logger.warning("[claw-eval] could not register rows in DatasetRegistry (non-fatal)", exc_info=True)

    return out

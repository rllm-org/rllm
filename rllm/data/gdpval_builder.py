"""Builder for the GDPval sandbox benchmark.

GDPval (``openai/gdpval``) is OpenAI's benchmark of real, economically valuable
knowledge work across 44 occupations in 9 GDP-heavy sectors. The open ("gold")
split ships 220 tasks; each row is::

    task_id, sector, occupation, prompt,
    reference_files[], reference_file_urls[], reference_file_hf_uris[],
    deliverable_files[], ...,          # the expert's reference deliverable
    rubric_pretty, rubric_json          # weighted pass/fail grading criteria

This module materializes the split into rLLM's ``type="sandbox"`` task-per-
directory layout so ``rllm eval gdpval`` can run each task in a sandbox and
grade the produced deliverable against ``rubric_json`` with the faithful
weighted-rubric LLM judge (:mod:`rllm.eval.reward_fns.gdpval`).

Used by ``rllm dataset pull gdpval`` (the ``builder`` field in
``rllm/registry/datasets.json`` → :func:`rllm.cli._pull.pull_dataset`).

On-disk output (``<out_dir>/``)::

    gdpval/
    ├── dataset.toml                       # [dataset] type="sandbox"
    ├── <task_id>/
    │   ├── task.toml                      # prompt/rubric_json in metadata + [verifier]
    │   ├── instruction.md                 # the role-first prompt + reference-file list
    │   ├── environment/files/             # reference files (uploaded to /workspace)
    │   └── tests/rubric.json              # structured rubric (debug/transparency)
    └── ...

Grading: each task's ``[verifier]`` points at ``gdpval_reward_fn``, a host-side
LLM-judge that reads the deliverable text surfaced on the episode and scores it
against the rubric. Surfacing the produced deliverable file to the episode
(``artifacts['deliverable_path']`` / ``['deliverable_text']``) is the harness
integration seam — see :func:`rllm.eval.reward_fns.gdpval._find_deliverable_text`.

NOTE: ``openai/gdpval`` reference files are pulled from the HF Hub; the full
220-task pull downloads all reference documents. Use ``--limit`` for smoke runs.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ID = "openai/gdpval"
# Default per-task verifiers, matched to the two catalog entries. The stamped
# [verifier] in task.toml MUST agree with the catalog reward_fn, or eval can
# grade with a different grader than intended depending on the dispatch path.
PAIRWISE_VERIFIER = "gdpval_pairwise_reward_fn"
RUBRIC_VERIFIER = "gdpval_reward_fn"


def _verifier_for(name: str, catalog_entry: dict | None) -> str:
    """The per-task verifier to stamp — the catalog reward_fn when known, else
    inferred from the dataset name (``*-rubric`` → rubric, else pairwise)."""
    rf = (catalog_entry or {}).get("reward_fn")
    if rf:
        return rf
    return RUBRIC_VERIFIER if "rubric" in name else PAIRWISE_VERIFIER


# Designated output directory (relative to the sandbox workdir) the agent is
# told to save deliverables into, and the surfacer reads first.
DELIVERABLE_DIR = "output"


def _toml_escape(s: str) -> str:
    """Escape a string for a TOML triple-quoted basic string."""
    return s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def _deliverable_basenames(row: dict) -> list[str]:
    """Expected deliverable filenames (basename) from the reference deliverables."""
    out = []
    for f in row.get("deliverable_files") or []:
        name = Path(str(f)).name
        if name:
            out.append(name)
    return out


def _write_instruction(task_dir: Path, row: dict, ref_names: list[str]) -> None:
    prompt = row.get("prompt") or ""
    lines = [prompt.strip(), ""]
    if ref_names:
        lines.append("## Reference files (available in your working directory)")
        lines += [f"- {n}" for n in ref_names]
        lines.append("")
    deliverables = _deliverable_basenames(row)
    if deliverables:
        lines.append("## Expected deliverable filename(s)")
        lines += [f"- {n}" for n in deliverables]
        lines.append("")
    # Harness instruction: a designated output directory removes any ambiguity
    # about which produced file is the deliverable to grade.
    lines.append("## Where to save your work")
    lines.append(f"Save your final deliverable file(s) into the `{DELIVERABLE_DIR}/` directory of your working directory. Only put the files you want graded there.")
    lines.append("")
    (task_dir / "instruction.md").write_text("\n".join(lines), encoding="utf-8")


def _write_task_toml(task_dir: Path, row: dict, ref_names: list[str], gold_names: list[str], verifier_name: str, judge_model: str | None) -> None:
    prompt = row.get("prompt") or ""
    rubric_json = row.get("rubric_json") or "[]"
    deliverables = _deliverable_basenames(row)

    # Top-level scalars land directly in Task.metadata (the loader copies the
    # whole task.toml into metadata), so the reward_fn can read prompt/rubric.
    # Scalars MUST precede the first [section] in TOML.
    lines = [
        f'task_id = "{row.get("task_id", "")}"',
        f'sector = """{_toml_escape(row.get("sector", ""))}"""',
        f'occupation = """{_toml_escape(row.get("occupation", ""))}"""',
        f'prompt = """{_toml_escape(prompt)}"""',
        f'rubric_json = """{_toml_escape(rubric_json)}"""',
        f"expected_deliverables = {json.dumps(deliverables)}",
        f"reference_files = {json.dumps(ref_names)}",
        # Gold (expert) deliverables staged under <task_dir>/reference/ — read by
        # the pairwise grader (gdpval_pairwise_reward_fn) via _find_reference_text.
        f"reference_deliverables = {json.dumps(gold_names)}",
        # Opt in to sandbox->host deliverable surfacing before host-side grading
        # (SandboxTaskHooks wraps the evaluator with _SurfacingEvaluator).
        "surface_deliverable = true",
        # Directory (under workdir) the agent is told to write deliverables to;
        # the surfacer reads only this directory.
        f'deliverable_dir = "{DELIVERABLE_DIR}"',
    ]
    if judge_model:
        lines.append(f'judge_model = "{judge_model}"')
    lines += [
        "",
        "[task]",
        f'name = "{row.get("task_id", "")}"',
        "",
        "[environment]",
        'workdir = "/workspace"',
        "",
        "[verifier]",
        f'name = "{verifier_name}"',
        "",
    ]
    (task_dir / "task.toml").write_text("\n".join(lines), encoding="utf-8")


def _stage_reference_files(row: dict, files_dir: Path) -> list[str]:
    """Download the task's reference files into ``files_dir``. Returns basenames.

    Uses ``hf_hub_download`` on the repo-relative paths in ``reference_files``
    (cached). Failures are logged and skipped so one bad file can't abort the
    whole build.
    """
    from huggingface_hub import hf_hub_download

    names: list[str] = []
    for rel in row.get("reference_files") or []:
        rel = str(rel)
        try:
            local = hf_hub_download(REPO_ID, rel, repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            logger.warning("[gdpval] could not download reference file %s: %s", rel, e)
            continue
        dest = files_dir / Path(rel).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
        names.append(dest.name)
    return names


def _stage_gold_deliverables(row: dict, ref_dir: Path) -> list[str]:
    """Download the task's gold (expert) deliverables into ``ref_dir``.

    These are the reference deliverables the pairwise grader compares against.
    Uses ``hf_hub_download`` on the repo-relative paths in ``deliverable_files``.
    """
    from huggingface_hub import hf_hub_download

    names: list[str] = []
    for rel in row.get("deliverable_files") or []:
        rel = str(rel)
        try:
            local = hf_hub_download(REPO_ID, rel, repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            logger.warning("[gdpval] could not download gold deliverable %s: %s", rel, e)
            continue
        dest = ref_dir / Path(rel).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
        names.append(dest.name)
    return names


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
    name: str = "gdpval",
    split: str = "train",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    limit: int | None = None,
    occupations: list[str] | None = None,
    default_agent: str = "claude-code",
    judge_model: str | None = None,
    skip_files: bool = False,
    clean: bool = False,
    register: bool = True,
) -> Path:
    """Materialize the GDPval split into a sandbox benchmark directory.

    Args:
        name: Dataset/registry name (also dataset.toml ``name``).
        split: HF split to build (``openai/gdpval`` publishes ``train``).
        out_dir: Output benchmark directory.
        catalog_entry: Optional datasets.json entry; ``description`` /
            ``default_agent`` / ``eval_split`` are read from it when present.
        limit: Keep only the first N tasks (after the occupation filter).
        occupations: Optional occupation allowlist (exact match), e.g. to build
            only the GDPval occupations that overlap the pipeline's coverage.
        default_agent: ``default_agent`` written into dataset.toml.
        judge_model: Optional judge model stamped into each task's metadata.
        skip_files: Skip downloading reference files (fast structural smoke run).
        clean: Remove ``out_dir`` before building.
        register: Also register rows in ``DatasetRegistry`` for ``rllm dataset`` parity.

    Returns:
        Path to the built benchmark directory.
    """
    from datasets import load_dataset

    if catalog_entry:
        split = catalog_entry.get("eval_split") or split
        default_agent = catalog_entry.get("default_agent") or default_agent

    out = Path(out_dir).expanduser()
    if clean and out.exists():
        logger.info("[gdpval] removing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("[gdpval] loading %s split=%s ...", REPO_ID, split)
    rows = [dict(r) for r in load_dataset(REPO_ID, split=split)]

    if occupations:
        keep = {o.strip().lower() for o in occupations}
        rows = [r for r in rows if str(r.get("occupation", "")).strip().lower() in keep]
    if limit is not None:
        rows = rows[:limit]
    logger.info("[gdpval] selected %d tasks (occupations=%s, limit=%s)", len(rows), occupations and len(occupations), limit)

    reg_rows: list[dict] = []
    n_with_files = 0
    n_with_gold = 0
    verifier_name = _verifier_for(name, catalog_entry)
    logger.info("[gdpval] per-task verifier: %s", verifier_name)
    for row in rows:
        task_id = row.get("task_id")
        if not task_id:
            continue
        task_dir = out / task_id
        if task_dir.exists():
            shutil.rmtree(task_dir)
        files_dir = task_dir / "environment" / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        ref_names = [] if skip_files else _stage_reference_files(row, files_dir)
        if ref_names:
            n_with_files += 1

        # Gold deliverables → <task_dir>/reference/ (used by the pairwise grader).
        gold_names = [] if skip_files else _stage_gold_deliverables(row, task_dir / "reference")
        if gold_names:
            n_with_gold += 1

        _write_instruction(task_dir, row, ref_names)
        _write_task_toml(task_dir, row, ref_names, gold_names, verifier_name, judge_model)

        # Ship the structured rubric alongside the task for transparency/debug.
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "rubric.json").write_text(row.get("rubric_json") or "[]", encoding="utf-8")

        reg_rows.append(
            {
                "task_id": task_id,
                "id": task_id,
                "instruction": (row.get("prompt") or "").strip(),
                "task_path": str(task_dir),
                "occupation": row.get("occupation", ""),
                "sector": row.get("sector", ""),
                "expected_deliverables": _deliverable_basenames(row),
                "reference_deliverables": gold_names,
            }
        )

    description = (catalog_entry or {}).get("description") or "GDPval (OpenAI): 220 gold tasks of economically valuable knowledge work (sandbox; weighted-rubric LLM-judge graded)."
    _write_dataset_toml(out, name=name, split=split, description=description, default_agent=default_agent)
    logger.info("[gdpval] wrote %d task dirs to %s (%d with reference files, %d with gold deliverables)", len(reg_rows), out, n_with_files, n_with_gold)

    if not reg_rows:
        raise RuntimeError(f"[gdpval] no tasks materialized from {REPO_ID} split={split}.")

    if register:
        try:
            from rllm.data import DatasetRegistry

            DatasetRegistry.register_dataset(
                name=name,
                data=reg_rows,
                split=split,
                source=REPO_ID,
                description=description,
                category=(catalog_entry or {}).get("category", "agentic"),
            )
        except Exception:  # registry parity is best-effort; eval uses the sandbox dir
            logger.warning("[gdpval] could not register rows in DatasetRegistry (non-fatal)", exc_info=True)

    return out


def main() -> None:
    """CLI: ``python -m rllm.data.gdpval_builder --out-dir <dir> [--limit N]``."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Materialize GDPval into an rLLM sandbox benchmark directory.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", default="gdpval")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--occupations", nargs="*", default=None)
    parser.add_argument("--default-agent", default="claude-code")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--skip-files", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=os.environ.get("RLLM_LOG_LEVEL", "INFO"))
    build_benchmark(
        name=args.name,
        split=args.split,
        out_dir=args.out_dir,
        limit=args.limit,
        occupations=args.occupations,
        default_agent=args.default_agent,
        judge_model=args.judge_model,
        skip_files=args.skip_files,
        clean=args.clean,
        register=False,
    )


if __name__ == "__main__":
    main()

"""Build the ``claw-eval-general`` sandbox benchmark from HuggingFace.

Claw-Eval (``claw-eval/Claw-Eval``) is a personal-assistant agent benchmark.
Each task is a *workspace*: a natural-language ``query`` plus a set of fixture
files (json/csv/docs the agent must read or act on). This script materializes
the 161-task ``general`` split into rLLM's ``type="sandbox"`` (task-per-
directory) layout so ``rllm eval`` can run each task in a Docker/Daytona
sandbox.

On-disk output (default ``~/.rllm/datasets/claw-eval-general``):

    claw-eval-general/
    ├── dataset.toml                     # [dataset] type="sandbox", default_agent="zeroclaw"
    ├── <task_id>/                       # one dir per row (e.g. T002_email_triage)
    │   ├── task.toml                    # top-level query/rubric + [environment]/[verifier]
    │   ├── instruction.md               # the row's query
    │   └── environment/files/fixtures/  # this task's fixtures (uploaded to /workspace)
    └── ...

Grading: the row ships no rubric, so each task points its ``[verifier]`` at the
``claw_eval_reward_fn`` LLM-judge (see rllm/eval/reward_fns/claw_eval.py), which
judges task completion from the agent's transcript.

Usage:
    uv run python scripts/data/claw_eval_dataset.py
    uv run python scripts/data/claw_eval_dataset.py --limit 10 --lang en --out ~/.rllm/datasets/claw-eval-general-10
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
from pathlib import Path

REPO_ID = "claw-eval/Claw-Eval"
VERIFIER_NAME = "claw_eval_reward_fn"


def _toml_escape(s: str) -> str:
    """Escape a string for a TOML triple-quoted basic string."""
    return s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def _write_task(
    task_dir: Path,
    *,
    task_id: str,
    query: str,
    category: str,
    language: str,
    judge_model: str | None,
) -> None:
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


def _write_dataset_toml(out: Path, *, name: str, split: str, default_agent: str) -> None:
    content = "\n".join(
        [
            "[dataset]",
            f'name = "{name}"',
            'type = "sandbox"',
            f'description = "Claw-Eval {split} split: personal-assistant agent tasks (LLM-judge graded)."',
            'default_sandbox = "docker"',
            f'default_agent = "{default_agent}"',
            f'split = "{split}"',
            "",
        ]
    )
    (out / "dataset.toml").write_text(content, encoding="utf-8")


def build(
    *,
    split: str,
    out: Path,
    limit: int | None,
    lang: str,
    default_agent: str,
    judge_model: str | None,
    clean: bool,
) -> None:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    if clean and out.exists():
        print(f"[claw-eval] removing existing {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[claw-eval] loading {REPO_ID} split={split} ...")
    ds = load_dataset(REPO_ID, split=split)

    rows = list(ds)
    if lang != "all":
        rows = [r for r in rows if r["language"] == lang]
    if limit is not None:
        rows = rows[:limit]
    selected_ids = {r["task_id"] for r in rows}
    print(f"[claw-eval] selected {len(rows)} tasks (lang={lang}, limit={limit})")

    # Extract only the fixtures for the selected tasks from the shared archive.
    print("[claw-eval] downloading fixtures archive (cached) ...")
    fixtures_path = hf_hub_download(REPO_ID, "data/fixtures.tar.gz", repo_type="dataset")

    # task_id -> destination environment/files dir
    dest_files = {r["task_id"]: out / r["task_id"] / "environment" / "files" for r in rows}
    extracted = {tid: 0 for tid in selected_ids}
    print("[claw-eval] extracting fixtures ...")
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

    # Write per-task task.toml + instruction.md.
    for r in rows:
        _write_task(
            out / r["task_id"],
            task_id=r["task_id"],
            query=r["query"],
            category=r["category"],
            language=r["language"],
            judge_model=judge_model,
        )

    _write_dataset_toml(out, name=out.name, split=split, default_agent=default_agent)

    n_with_fx = sum(1 for tid in selected_ids if extracted[tid] > 0)
    n_empty = len(selected_ids) - n_with_fx
    print(f"[claw-eval] wrote {len(rows)} task dirs to {out}")
    print(f"[claw-eval]   {n_with_fx} with fixtures, {n_empty} fixture-less (web/research tasks)")
    print(f"[claw-eval] done. Run:  rllm eval {out} --agent {default_agent}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the claw-eval sandbox benchmark.")
    ap.add_argument("--split", default="general", help="HF split (default: general)")
    ap.add_argument("--out", default=None, help="output dir (default: ~/.rllm/datasets/claw-eval-<split>)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks (after lang filter)")
    ap.add_argument("--lang", default="all", choices=["all", "en", "zh"], help="language filter")
    ap.add_argument("--default-agent", default="zeroclaw", help="default_agent in dataset.toml")
    ap.add_argument("--judge-model", default=None, help="override judge model stamped into each task")
    ap.add_argument("--clean", action="store_true", help="remove the output dir first")
    args = ap.parse_args()

    out = Path(args.out).expanduser() if args.out else Path(os.path.expanduser(f"~/.rllm/datasets/claw-eval-{args.split}"))
    build(
        split=args.split,
        out=out,
        limit=args.limit,
        lang=args.lang,
        default_agent=args.default_agent,
        judge_model=args.judge_model,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()

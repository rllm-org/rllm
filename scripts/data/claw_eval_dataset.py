"""Build the Claw-Eval sandbox benchmark from HuggingFace (standalone CLI).

Thin wrapper over :func:`rllm.data.claw_eval_builder.build_benchmark` — the
same builder used by ``rllm dataset pull claw_eval``. Prefer the CLI for normal
use; this script exists for ad-hoc subsets and custom output paths.

Usage:
    uv run python scripts/data/claw_eval_dataset.py                       # full general split
    uv run python scripts/data/claw_eval_dataset.py --lang en --limit 10 \
        --out ~/.rllm/datasets/claw-eval-general-10
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from rllm.data.claw_eval_builder import build_benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Claw-Eval sandbox benchmark.")
    ap.add_argument("--split", default="general", help="HF split (default: general)")
    ap.add_argument("--out", default=None, help="output dir (default: ~/.rllm/datasets/claw-eval-<split>)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks (after lang filter)")
    ap.add_argument("--lang", default="all", choices=["all", "en", "zh"], help="language filter")
    ap.add_argument("--name", default="claw_eval", help="dataset name written into dataset.toml")
    ap.add_argument("--default-agent", default="zeroclaw", help="default_agent in dataset.toml")
    ap.add_argument("--judge-model", default=None, help="override judge model stamped into each task")
    ap.add_argument("--clean", action="store_true", help="remove the output dir first")
    ap.add_argument("--no-register", action="store_true", help="don't register rows in DatasetRegistry")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(args.out).expanduser() if args.out else Path(os.path.expanduser(f"~/.rllm/datasets/claw-eval-{args.split}"))
    build_benchmark(
        name=args.name,
        split=args.split,
        out_dir=out,
        limit=args.limit,
        lang=args.lang,
        default_agent=args.default_agent,
        judge_model=args.judge_model,
        clean=args.clean,
        register=not args.no_register,
    )
    print(f"Done. Run:  rllm eval {out} --agent {args.default_agent}")


if __name__ == "__main__":
    main()

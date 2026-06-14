"""Pull the train + eval datasets for the swe-rl cookbook.

Both are sandbox-format benchmarks (per-task ``environment/Dockerfile`` +
``tests/test.sh`` verifier). The training set is rLLM's native
``rllm-swesmith`` (filtered SWE-smith, ~4.7K bug-fix tasks across 105
Python repos). The eval set is ``harbor:swebench-verified`` (500
real-world GitHub issues, evaluated against the official SWE-bench
harness inside the sandbox).

This script is a thin wrapper around ``rllm dataset pull`` so the
cookbook can be used end-to-end with a single command. Re-runs are
no-ops once the on-disk benchmark directory exists.

Usage::

    python cookbooks/swe-rl/prepare_data.py
    # or, smoke run with a small training cap (rebuilds the benchmark dir):
    python cookbooks/swe-rl/prepare_data.py --train-limit 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys

TRAIN_DATASET = "rllm-swesmith"
VAL_DATASET = "harbor:swebench-verified"


def _pull(name: str, limit: int | None = None) -> None:
    cmd = [sys.executable, "-m", "rllm.cli.main", "dataset", "pull", name]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    print(f"[swe-rl] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Cap training tasks (default: full ~4.7K). Useful for smoke runs.",
    )
    ap.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Cap eval tasks (default: full 500).",
    )
    args = ap.parse_args()

    _pull(TRAIN_DATASET, limit=args.train_limit)
    _pull(VAL_DATASET, limit=args.val_limit)

    print(
        f"\n[swe-rl] Done. Train: {TRAIN_DATASET}   Eval: {VAL_DATASET}\n        Run `bash cookbooks/swe-rl/train_tinker.sh` to train.",
        flush=True,
    )


if __name__ == "__main__":
    main()

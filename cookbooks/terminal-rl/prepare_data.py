"""Pull the train + eval datasets for the terminal-rl cookbook.

Both are sandbox-format benchmarks (per-task ``environment/Dockerfile`` +
``tests/test.sh`` verifier) in Harbor's directory-per-task layout. The two
sides differ only in *where* the tasks come from:

* **Train** — a local tarball of Harbor-format task directories that you
  provide (path set via ``TB_TRAIN_TARBALL`` or ``--tarball``). The tarball is
  extracted once under the rLLM datasets dir and each task directory is
  converted into a flat row (``task_path`` + ``instruction``) and registered as
  the ``tb-opus-pass`` dataset's ``train`` split. Each task carries its own
  prebuilt ``docker_image`` and a ``tests/test.sh`` that writes
  ``/logs/verifier/reward.txt`` — exactly the signal rLLM's per-task verifier
  reads back.
* **Eval** — ``harbor:terminal-bench@<version>`` pulled straight from the
  Harbor registry (the same path the Terminal-Bench eval cookbook uses).
  ``TB_EVAL_VERSION`` selects the version (default ``2.0``; the registry only
  publishes ``2.0`` today, so set ``TB_EVAL_VERSION=2.1`` once it lands).

Re-runs are cheap: extraction is skipped once the on-disk task tree exists,
and the eval pull is a no-op once the tasks are cached locally.

Usage::

    python cookbooks/terminal-rl/prepare_data.py
    # smoke run with a small training cap (still extracts the full tarball):
    python cookbooks/terminal-rl/prepare_data.py --train-limit 50
    # evaluate against a different Terminal-Bench version:
    TB_EVAL_VERSION=2.1 python cookbooks/terminal-rl/prepare_data.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Registry name for the local training set.
TRAIN_DATASET = "tb-opus-pass"
TRAIN_SPLIT = "train"

# Terminal-Bench eval version (Harbor registry). 2.0 is what the registry
# publishes today; flip to 2.1 (or any published version) via TB_EVAL_VERSION.
EVAL_VERSION = os.environ.get("TB_EVAL_VERSION", "2.0")
EVAL_DATASET = f"terminal-bench@{EVAL_VERSION}"

# Local training tarball (Harbor tasks). Override with TB_TRAIN_TARBALL.
DEFAULT_TARBALL = os.path.expanduser(os.environ.get("TB_TRAIN_TARBALL", "~/terminal_train_tasks.tar.zst"))


def _tasks_root() -> Path:
    """Where the training tarball is extracted (under the rLLM datasets home)."""
    from rllm import paths

    return Path(paths.datasets_dir()) / TRAIN_DATASET / "tasks"


def _extract_tarball(tarball: Path, dest: Path) -> Path:
    """Extract the ``.tar.zst`` tarball into ``dest`` (idempotent).

    Returns the directory whose immediate children are Harbor task dirs
    (unwrapping a single top-level wrapper dir if the archive has one).
    """
    marker = dest / ".extracted"
    if not marker.exists():
        if not tarball.exists():
            raise FileNotFoundError(f"Training tarball not found: {tarball}\nSet TB_TRAIN_TARBALL to its path, e.g. TB_TRAIN_TARBALL=/path/to/your_tasks.tar.zst")
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[terminal-rl] Extracting {tarball} -> {dest} (one-time)...", flush=True)
        # --use-compress-program=unzstd works without GNU tar's --zstd support
        # and without a Python zstandard dependency.
        subprocess.run(
            ["tar", "--use-compress-program=unzstd", "-xf", str(tarball), "-C", str(dest)],
            check=True,
        )
        marker.touch()
    else:
        print(f"[terminal-rl] Reusing extracted tasks at {dest}", flush=True)

    # Unwrap a single top-level wrapper directory (if the tarball ships one);
    # fall back to ``dest`` if the task dirs sit directly under it.
    children = [d for d in dest.iterdir() if d.is_dir()]
    if len(children) == 1 and not (children[0] / "task.toml").exists():
        return children[0]
    return dest


def _register_train(tasks_root: Path, limit: int | None) -> int:
    """Convert each Harbor task dir into a row and register the train split."""
    from rllm.data import DatasetRegistry
    from rllm.integrations.harbor.dataset_loader import harbor_task_to_row

    task_dirs = sorted(d for d in tasks_root.iterdir() if d.is_dir() and (d / "task.toml").exists())
    if not task_dirs:
        raise RuntimeError(f"No Harbor task directories (task.toml) found under {tasks_root}")
    if limit is not None:
        task_dirs = task_dirs[:limit]

    rows = [row for d in task_dirs if (row := harbor_task_to_row(d)) is not None]
    if not rows:
        raise RuntimeError(f"All {len(task_dirs)} task dirs under {tasks_root} were invalid/skipped")

    DatasetRegistry.register_dataset(
        name=TRAIN_DATASET,
        data=rows,
        split=TRAIN_SPLIT,
        source=f"local:{Path(DEFAULT_TARBALL).name}",
        description="Local terminal-agent tasks (Harbor format; per-task tests/test.sh verifier)",
        category="agentic",
    )
    return len(rows)


def _pull_eval() -> None:
    """Pull the Terminal-Bench eval split from the Harbor registry."""
    name = f"harbor:{EVAL_DATASET}"
    cmd = [sys.executable, "-m", "rllm.cli.main", "dataset", "pull", name]
    print(f"[terminal-rl] $ {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"[terminal-rl] Failed to pull '{name}'. The Harbor registry currently "
            f"publishes terminal-bench@2.0; if you requested a version it does not "
            f"have, set TB_EVAL_VERSION to an available one (e.g. 2.0)."
        ) from e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Cap training tasks (default: all ~12.6K). Useful for smoke runs.",
    )
    ap.add_argument(
        "--tarball",
        type=str,
        default=DEFAULT_TARBALL,
        help=f"Path to the training tarball (default: {DEFAULT_TARBALL}).",
    )
    args = ap.parse_args()

    tasks_root = _extract_tarball(Path(args.tarball).expanduser(), _tasks_root())
    n_train = _register_train(tasks_root, args.train_limit)
    print(f"[terminal-rl] Registered {TRAIN_DATASET}/{TRAIN_SPLIT} ({n_train} tasks)", flush=True)

    _pull_eval()

    print(
        f"\n[terminal-rl] Done. Train: {TRAIN_DATASET} ({n_train})   Eval: {EVAL_DATASET}\n        Run `bash cookbooks/terminal-rl/train_tinker.sh` to train.",
        flush=True,
    )


if __name__ == "__main__":
    main()

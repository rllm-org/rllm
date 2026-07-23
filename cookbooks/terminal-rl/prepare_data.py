"""Pull the train + eval datasets for the terminal-rl cookbook.

Both are sandbox-format benchmarks (per-task ``environment/Dockerfile`` +
``tests/test.sh`` verifier) in Harbor's directory-per-task layout. The two
sides differ only in *where* the tasks come from:

* **Train** — a local ``.tar.zst`` or ``.zip`` archive of Harbor-format task
  directories that you provide (path set via ``TB_TRAIN_TARBALL`` or
  ``--tarball``). The archive is
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
    # register the standalone eight-task debug archive without touching train:
    python cookbooks/terminal-rl/prepare_data.py \
        --debug-only --tarball /path/to/tb_v2_debug_tasks.tar.zst
    # smoke run with a small training cap (still extracts the full archive):
    python cookbooks/terminal-rl/prepare_data.py --train-limit 50
    # evaluate against a different Terminal-Bench version:
    TB_EVAL_VERSION=2.1 python cookbooks/terminal-rl/prepare_data.py
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

# Registry name for the local training set.
TRAIN_DATASET = "tb-opus-pass"
TRAIN_SPLIT = "train"

DEBUG_DATASET = "tb_v2_debug"
DEBUG_TASKS = (
    "bottleneck-path-oracle",
    "debug-rl-library",
    "ecc-curve-audit",
    "erniekit-config-validator",
    "fix-numerical-bugs",
    "maxwell-cavity-modes",
    "mx-format-gemm",
    "parametric-qp-breakpoints",
)

# Terminal-Bench eval version (Harbor registry). 2.0 is what the registry
# publishes today; flip to 2.1 (or any published version) via TB_EVAL_VERSION.
EVAL_VERSION = os.environ.get("TB_EVAL_VERSION", "2.0")
EVAL_DATASET = f"terminal-bench@{EVAL_VERSION}"

# Local training tarball (Harbor tasks). Override with TB_TRAIN_TARBALL.
DEFAULT_TARBALL = os.path.expanduser(
    os.environ.get(
        "TB_TRAIN_TARBALL",
        str(Path(__file__).resolve().parent / "tb-v2-tasks" / "tb_v2_tasks.tar.zst"),
    )
)


def _tasks_root() -> Path:
    """Where the training tarball is extracted (under the rLLM datasets home)."""
    from rllm import paths

    return Path(paths.datasets_dir()) / TRAIN_DATASET / "tasks"


def _debug_tasks_root() -> Path:
    """Where a standalone debug tarball is extracted."""
    from rllm import paths

    return Path(paths.datasets_dir()) / DEBUG_DATASET / "tasks"


def _extract_archive(archive: Path, dest: Path) -> Path:
    """Extract a ``.tar.zst`` or ``.zip`` archive into ``dest`` (idempotent).

    Returns the directory whose immediate children are Harbor task dirs
    (unwrapping a single top-level wrapper dir if the archive has one). macOS
    ZIP metadata directories are ignored when deciding whether to unwrap.
    """
    marker = dest / ".extracted"
    if not marker.exists():
        if not archive.exists():
            raise FileNotFoundError(f"Training archive not found: {archive}\nSet TB_TRAIN_TARBALL to its path, e.g. TB_TRAIN_TARBALL=/path/to/your_tasks.zip")
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[terminal-rl] Extracting {archive} -> {dest} (one-time)...", flush=True)
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive) as zf:
                for member in zf.infolist():
                    path = PurePosixPath(member.filename)
                    mode = member.external_attr >> 16
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError(f"Unsafe ZIP member path: {member.filename!r}")
                    if stat.S_ISLNK(mode):
                        raise ValueError(f"ZIP symlinks are not supported: {member.filename!r}")
                    if member.flag_bits & 0x1:
                        raise ValueError(f"Encrypted ZIP members are not supported: {member.filename!r}")
                zf.extractall(dest)
        else:
            # --use-compress-program=unzstd works without GNU tar's --zstd
            # support and without a Python zstandard dependency.
            subprocess.run(
                [
                    "tar",
                    "--use-compress-program=unzstd",
                    "--no-same-owner",
                    "--no-same-permissions",
                    "-xf",
                    str(archive),
                    "-C",
                    str(dest),
                ],
                check=True,
            )
        marker.touch()
    else:
        print(f"[terminal-rl] Reusing extracted tasks at {dest}", flush=True)

    # Unwrap a single top-level wrapper directory (if the tarball ships one);
    # fall back to ``dest`` if the task dirs sit directly under it. Finder adds
    # ``__MACOSX`` siblings to ZIPs; those are metadata, not task roots.
    children = [d for d in dest.iterdir() if d.is_dir() and d.name != "__MACOSX" and not d.name.startswith(".")]
    if len(children) == 1 and not (children[0] / "task.toml").exists():
        return children[0]
    return dest


def _register_train(tasks_root: Path, limit: int | None, tarball: Path) -> int:
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
        source=f"local:{tarball.name}",
        description="Local terminal-agent tasks (Harbor format; per-task tests/test.sh verifier)",
        category="agentic",
    )
    return len(rows)


def _register_debug(tasks_root: Path, tarball: Path) -> int:
    """Register the selected eight-task debug subset from the training archive."""
    from rllm.data import DatasetRegistry
    from rllm.integrations.harbor.dataset_loader import harbor_task_to_row

    task_dirs = {d.name: d for d in tasks_root.iterdir() if d.is_dir() and (d / "task.toml").exists()}
    missing = [name for name in DEBUG_TASKS if name not in task_dirs]
    if missing:
        missing_names = ", ".join(missing)
        raise RuntimeError(f"Training tarball is missing debug tasks: {missing_names}")

    rows = []
    invalid = []
    for name in DEBUG_TASKS:
        row = harbor_task_to_row(task_dirs[name])
        if row is None:
            invalid.append(name)
        else:
            rows.append(row)
    if invalid:
        invalid_names = ", ".join(invalid)
        raise RuntimeError(f"Invalid debug tasks: {invalid_names}")

    DatasetRegistry.register_dataset(
        name=DEBUG_DATASET,
        data=rows,
        split=TRAIN_SPLIT,
        source=f"local:{tarball.name}",
        description="Eight-task Terminal-Bench v2 debug subset",
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
        help="Cap training tasks (default: all tasks in the archive). Useful for smoke runs.",
    )
    ap.add_argument(
        "--tarball",
        type=str,
        default=DEFAULT_TARBALL,
        help=f"Path to the training .tar.zst or .zip archive (default: {DEFAULT_TARBALL}).",
    )
    ap.add_argument(
        "--debug-only",
        action="store_true",
        help=("Register a standalone eight-task debug archive as tb_v2_debug without creating or replacing tb-opus-pass."),
    )
    args = ap.parse_args()

    if args.debug_only and args.train_limit is not None:
        ap.error("--train-limit cannot be used with --debug-only")

    tarball = Path(args.tarball).expanduser()
    if args.debug_only:
        tasks_root = _extract_archive(tarball, _debug_tasks_root())
        n_train = None
    else:
        tasks_root = _extract_archive(tarball, _tasks_root())
        n_train = _register_train(tasks_root, args.train_limit, tarball)
        print(f"[terminal-rl] Registered {TRAIN_DATASET}/{TRAIN_SPLIT} ({n_train} tasks)", flush=True)

    if args.debug_only:
        n_debug = _register_debug(tasks_root, tarball)
        print(f"[terminal-rl] Registered {DEBUG_DATASET}/{TRAIN_SPLIT} ({n_debug} tasks)", flush=True)
    else:
        task_names = {d.name for d in tasks_root.iterdir() if d.is_dir() and (d / "task.toml").exists()}
        if set(DEBUG_TASKS).issubset(task_names):
            n_debug = _register_debug(tasks_root, tarball)
            print(f"[terminal-rl] Registered {DEBUG_DATASET}/{TRAIN_SPLIT} ({n_debug} tasks)", flush=True)
        else:
            n_debug = None
            print(
                "[terminal-rl] Full archive does not contain the standalone eight-task debug set; leaving any existing tb_v2_debug registration unchanged.",
                flush=True,
            )

    _pull_eval()

    if args.debug_only:
        summary = f"Debug: {DEBUG_DATASET} ({n_debug})   Eval: {EVAL_DATASET}"
    else:
        summary = f"Train: {TRAIN_DATASET} ({n_train})   Eval: {EVAL_DATASET}"
    print(f"\n[terminal-rl] Done. {summary}", flush=True)


if __name__ == "__main__":
    main()

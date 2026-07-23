"""Pull the train + eval datasets for the terminal-rl cookbook.

Both are sandbox-format benchmarks (per-task ``environment/Dockerfile`` +
``tests/test.sh`` verifier) in Harbor's directory-per-task layout. The two
sides differ only in *where* the tasks come from:

* **Train** — a local ``.tar.zst`` or ``.zip`` archive of Harbor-format task
  directories that you provide (path set via ``TB_TRAIN_TARBALL`` or
  ``--tarball``). The archive is
  extracted once under the rLLM datasets dir and each task directory is
  converted into a flat row (``task_path`` + ``instruction``) and registered
  as ``tb-opus-pass/train``. Each task carries its own
  prebuilt ``docker_image`` and a ``tests/test.sh`` that writes
  ``/logs/verifier/reward.txt`` — exactly the signal rLLM's per-task verifier
  reads back.
* **Benchmarks** — Terminal-Bench 2.0 is retained for the existing debug and
  comparison profiles. The 89-task Terminal-Bench 2.1 package is pulled from
  the Harbor registry at an immutable revision and registered locally as
  ``terminal-bench@2.1/default``. A deterministic eight-task subset is also
  registered as ``terminal-bench@2.1/midtest`` for periodic production
  evaluation.

Re-runs are cheap: extraction is skipped once the on-disk task tree exists,
and the eval pull is a no-op once the tasks are cached locally.

Usage::

    python cookbooks/terminal-rl/prepare_data.py
    # register the standalone eight-task debug archive without touching train:
    python cookbooks/terminal-rl/prepare_data.py \
        --debug-only --tarball /path/to/tb_v2_debug_tasks.tar.zst
    # smoke run with a small training cap (still extracts the full archive):
    python cookbooks/terminal-rl/prepare_data.py --train-limit 50
    # change the deterministic Terminal-Bench 2.1 mid-test subset:
    python cookbooks/terminal-rl/prepare_data.py \
        --midtest-size 8 --midtest-seed 20260723
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import subprocess
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

LEGACY_EVAL_DATASET = "terminal-bench@2.0"
LEGACY_EVAL_SOURCE = "terminal-bench@2.0"
LEGACY_EVAL_EXPECTED_TASKS = 89

# Terminal-Bench 2.1 currently resolves to package revision 6 (89 tasks,
# content hash 7d7bdc1c...a0699a). Pinning the immutable revision prevents a
# mutable "latest" tag from changing the step-0/final comparison mid-project.
EVAL_DATASET = os.environ.get("TB_BENCHMARK_DATASET", "terminal-bench@2.1")
EVAL_SOURCE = os.environ.get(
    "TB_BENCHMARK_SOURCE",
    "terminal-bench/terminal-bench-2-1@6",
)
EVAL_EXPECTED_TASKS = int(os.environ.get("TB_BENCHMARK_EXPECTED_TASKS", "89"))
MIDTEST_SPLIT = "midtest"
DEFAULT_MIDTEST_SIZE = int(os.environ.get("TB_MIDTEST_SIZE", "8"))
DEFAULT_MIDTEST_SEED = int(os.environ.get("TB_MIDTEST_SEED", "20260723"))

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


def _select_fixed_subset(rows: list[dict], subset_size: int, subset_seed: int) -> list[dict]:
    """Select a stable subset by seeded task-id hash."""
    if not 0 < subset_size <= len(rows):
        raise ValueError(f"subset size must be between 1 and {len(rows)}, got {subset_size}")

    task_ids = [str(row.get("task_id", "")) for row in rows]
    if any(not task_id for task_id in task_ids):
        raise ValueError("every row must have a non-empty task_id")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_id values must be unique before selecting a fixed subset")

    ranked_ids = sorted(
        task_ids,
        key=lambda task_id: (
            hashlib.sha256(f"{subset_seed}:{task_id}".encode()).digest(),
            task_id,
        ),
    )
    selected_ids = set(ranked_ids[:subset_size])
    return [row for row in rows if str(row["task_id"]) in selected_ids]


def _register_train(tasks_root: Path, limit: int | None, tarball: Path) -> int:
    """Convert Harbor tasks and register the complete internal training split."""
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


def _pull_eval(midtest_size: int, midtest_seed: int) -> tuple[int, int]:
    """Register the full pinned benchmark plus a fixed periodic subset."""
    from rllm.data import DatasetRegistry
    from rllm.integrations.harbor.dataset_loader import load_harbor_dataset

    info = DatasetRegistry.get_dataset_info(EVAL_DATASET)
    rows = None
    if info is not None:
        split = info.get("splits", {}).get("default", {})
        source = info.get("metadata", {}).get("source")
        if split.get("num_examples") == EVAL_EXPECTED_TASKS and source == f"harbor:{EVAL_SOURCE}":
            dataset = DatasetRegistry.load_dataset(EVAL_DATASET, "default")
            rows = dataset.get_data() if dataset is not None else None
            print(
                f"[terminal-rl] Reusing {EVAL_DATASET}/default ({EVAL_EXPECTED_TASKS} tasks, source {EVAL_SOURCE})",
                flush=True,
            )

    if rows is None:
        print(f"[terminal-rl] Pulling Harbor benchmark {EVAL_SOURCE}", flush=True)
        rows = load_harbor_dataset(EVAL_SOURCE)
        if len(rows) != EVAL_EXPECTED_TASKS:
            raise RuntimeError(f"{EVAL_SOURCE} resolved to {len(rows)} tasks; expected exactly {EVAL_EXPECTED_TASKS}")

        DatasetRegistry.register_dataset(
            name=EVAL_DATASET,
            data=rows,
            split="default",
            source=f"harbor:{EVAL_SOURCE}",
            description="Pinned Terminal-Bench 2.1 boundary benchmark",
            category="agentic",
        )

    midtest_rows = _select_fixed_subset(rows, midtest_size, midtest_seed)
    DatasetRegistry.register_dataset(
        name=EVAL_DATASET,
        data=midtest_rows,
        split=MIDTEST_SPLIT,
        source=f"harbor:{EVAL_SOURCE}",
        description=(f"Pinned Terminal-Bench 2.1 boundary benchmark with deterministic {midtest_size}-task mid-test subset (seed {midtest_seed})"),
        category="agentic",
    )
    return len(rows), len(midtest_rows)


def _pull_legacy_eval() -> int:
    """Retain Terminal-Bench 2.0 for the debug and four-run profiles."""
    from rllm.data import DatasetRegistry
    from rllm.integrations.harbor.dataset_loader import load_harbor_dataset

    info = DatasetRegistry.get_dataset_info(LEGACY_EVAL_DATASET)
    if info is not None:
        split = info.get("splits", {}).get("default", {})
        source = info.get("metadata", {}).get("source")
        if split.get("num_examples") == LEGACY_EVAL_EXPECTED_TASKS and source == f"harbor:{LEGACY_EVAL_SOURCE}":
            print(
                f"[terminal-rl] Reusing {LEGACY_EVAL_DATASET}/default ({LEGACY_EVAL_EXPECTED_TASKS} tasks)",
                flush=True,
            )
            return LEGACY_EVAL_EXPECTED_TASKS

    print(f"[terminal-rl] Pulling Harbor benchmark {LEGACY_EVAL_SOURCE}", flush=True)
    rows = load_harbor_dataset(LEGACY_EVAL_SOURCE)
    if len(rows) != LEGACY_EVAL_EXPECTED_TASKS:
        raise RuntimeError(f"{LEGACY_EVAL_SOURCE} resolved to {len(rows)} tasks; expected exactly {LEGACY_EVAL_EXPECTED_TASKS}")

    DatasetRegistry.register_dataset(
        name=LEGACY_EVAL_DATASET,
        data=rows,
        split="default",
        source=f"harbor:{LEGACY_EVAL_SOURCE}",
        description="Terminal-Bench 2.0 comparison benchmark",
        category="agentic",
    )
    return len(rows)


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
    ap.add_argument(
        "--midtest-size",
        type=int,
        default=DEFAULT_MIDTEST_SIZE,
        help=(f"Number of pinned Terminal-Bench 2.1 tasks registered as {EVAL_DATASET}/{MIDTEST_SPLIT} (default: {DEFAULT_MIDTEST_SIZE})."),
    )
    ap.add_argument(
        "--midtest-seed",
        type=int,
        default=DEFAULT_MIDTEST_SEED,
        help=f"Seed for deterministic mid-test task selection (default: {DEFAULT_MIDTEST_SEED}).",
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

    n_legacy_eval = _pull_legacy_eval()
    n_eval, n_midtest = _pull_eval(args.midtest_size, args.midtest_seed)

    if args.debug_only:
        summary = (
            f"Debug: {DEBUG_DATASET} ({n_debug})   "
            f"Legacy eval: {LEGACY_EVAL_DATASET}/default ({n_legacy_eval})   "
            f"Mid-test: {EVAL_DATASET}/{MIDTEST_SPLIT} ({n_midtest})   "
            f"Benchmark: {EVAL_DATASET}/default ({n_eval})"
        )
    else:
        summary = (
            f"Train: {TRAIN_DATASET}/{TRAIN_SPLIT} ({n_train})   "
            f"Legacy eval: {LEGACY_EVAL_DATASET}/default ({n_legacy_eval})   "
            f"Mid-test: {EVAL_DATASET}/{MIDTEST_SPLIT} ({n_midtest})   "
            f"Benchmark: {EVAL_DATASET}/default ({n_eval})"
        )
    print(f"\n[terminal-rl] Done. {summary}", flush=True)


if __name__ == "__main__":
    main()

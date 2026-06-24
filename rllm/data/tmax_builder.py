"""Builder for Ai2's TMax-15K terminal-agent RL corpus.

TMax-15K (arXiv:2606.23321) is ~14.6K compositional terminal-agent tasks. Ai2
publishes it on the Hugging Face Hub in two complementary forms, which this
builder joins by ``task_id``:

* ``allenai/TMax-15K`` (raw) carries the **programmatic verifier**
  ``test_final_state`` (a pytest module whose pass/fail IS the RL reward — see
  the project's ``rl_data/README.md``: "test_final_state.py = programmatic
  verifier (pass/fail signal)") plus the task ``description`` and the
  ``container_def`` the environment was built from.
* ``allenai/tmax-15k-open-instruct`` (training-ready) carries the **per-task
  prebuilt Docker image** (``env_config.image`` = ``hamishi740/swerl-tmax-v3:<digest>``),
  so we pull a ready environment instead of building from ``container_def``.

The Ai2 README claims the corpus is also on the Harbor registry as
``tmax/TMax-15K-Harbor``, but that dataset is not present in the public Harbor
package registry (``list_datasets()`` returns 80 datasets, none tmax), so this
builder sources from Hugging Face instead.

Each row becomes a Harbor-format task directory whose verifier runs the task's
``test_final_state.py`` with pytest inside the prebuilt image; reward = 1.0 iff
it passes. This mirrors ``r2egym_builder`` / ``swesmith_builder``.

On-disk output (``<out_dir>/``)::

    tmax-15k/
    ├── dataset.toml                       # type="sandbox"
    ├── <task_id>/
    │   ├── task.toml                      # docker_image=<prebuilt image>
    │   ├── instruction.md                 # task description
    │   ├── environment/Dockerfile         # FROM <image> + ENTRYPOINT []
    │   └── tests/
    │       ├── test.sh                    # run test_final_state.py, write reward
    │       └── test_final_state.py        # the programmatic verifier (from raw)
    └── ...

Invoked from ``rllm dataset pull tmax-15k`` via the ``builder`` field in
``rllm/registry/datasets.json`` → :func:`rllm.cli._pull.pull_dataset`.

NOTE: pulling the per-task images at the full 15K scale needs a Docker Hub
business account (rate limits); small subsets (``--limit``) pull fine.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Raw corpus: carries test_final_state (verifier) + description.
DEFAULT_HF_REPO_ID = "allenai/TMax-15K"
# Training-ready corpus: carries the per-task prebuilt docker image.
IMAGE_HF_REPO_ID = "allenai/tmax-15k-open-instruct"

# Per-instance resource defaults. The swerl images bundle a full task
# environment (compilers, services, oracle binaries for graded verifiers).
_DEFAULT_RESOURCES = {
    "cpus": 4,
    "memory_mb": 8192,
    "storage_mb": 20480,
    "build_timeout_sec": 1800.0,
}
_DEFAULT_TIMEOUTS = {
    "agent_timeout_sec": 1800.0,
    "verifier_timeout_sec": 300.0,
}


def _load_rows(hf_repo_id: str, hf_split: str, *, retries: int = 4, backoff_sec: float = 10.0) -> list[dict]:
    """Load HF rows with retries on transient Hub errors (mirrors r2egym_builder)."""
    from datasets import load_dataset

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            ds = load_dataset(hf_repo_id, split=hf_split)
            return [dict(r) for r in ds]
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt == retries:
                break
            wait = backoff_sec * attempt
            logger.warning("[tmax] load_dataset(%s) failed (attempt %d/%d): %s — retry in %.0fs", hf_repo_id, attempt, retries, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"load_dataset({hf_repo_id!r}, split={hf_split!r}) failed after {retries} attempts") from last_exc


def _load_image_map(image_repo_id: str, hf_split: str) -> dict[str, str]:
    """Map ``task_id`` → prebuilt docker image from the open-instruct dataset."""
    rows = _load_rows(image_repo_id, hf_split)
    image_map: dict[str, str] = {}
    for r in rows:
        env = r.get("env_config") or {}
        tid = env.get("task_id")
        img = env.get("image")
        if tid and img:
            image_map[tid] = img
    logger.info("[tmax] loaded %d task→image mappings from %s", len(image_map), image_repo_id)
    return image_map


def _build_dockerfile(docker_image: str) -> str:
    """``FROM <image>`` + clear ENTRYPOINT so the sandbox can exec into it."""
    return f"FROM {docker_image}\nENTRYPOINT []\n"


def _build_task_toml(*, task_id: str, domain: str, docker_image: str) -> str:
    """Synthesize a Harbor-format ``task.toml``."""
    safe_domain = (domain or "").replace('"', "'")
    lines = [
        'schema_version = "1.1"',
        "",
        "[task]",
        f'name = "tmax-15k/{task_id}"',
        f'description = "TMax-15K task ({safe_domain})"',
        'keywords = ["tmax", "terminal-agent"]',
        "",
        "[metadata]",
        f'task_id = "{task_id}"',
        f'domain = "{safe_domain}"',
        "",
        "[environment]",
        f'docker_image = "{docker_image}"',
        f"cpus = {_DEFAULT_RESOURCES['cpus']}",
        f"memory_mb = {_DEFAULT_RESOURCES['memory_mb']}",
        f"storage_mb = {_DEFAULT_RESOURCES['storage_mb']}",
        f"build_timeout_sec = {_DEFAULT_RESOURCES['build_timeout_sec']}",
        "allow_internet = true",
        "",
        "[agent]",
        f"timeout_sec = {_DEFAULT_TIMEOUTS['agent_timeout_sec']}",
        "",
        "[verifier]",
        f"timeout_sec = {_DEFAULT_TIMEOUTS['verifier_timeout_sec']}",
        "",
    ]
    return "\n".join(lines)


# Reward = 1.0 iff the task's programmatic verifier (test_final_state.py) passes
# under pytest, run inside the prebuilt image. pytest exit 0 = all collected
# tests passed; exit 5 (no tests collected) or any failure ⇒ 0.0. The reward
# file path matches rllm.eval.script_evaluator's search order (/tmp/rllm/reward.json
# wins). The verifier file ships in the task's tests/ dir, available at
# /tests/test_final_state.py inside the sandbox.
_VERIFIER_TEMPLATE = r"""#!/bin/bash
set -uo pipefail
mkdir -p /tmp/rllm /logs/verifier
log() { echo "[verifier] $*"; }

PY=python3; command -v "$PY" >/dev/null 2>&1 || PY=python
# test_final_state.py is a pytest module; ensure pytest is importable.
if ! "$PY" -m pytest --version >/dev/null 2>&1; then
    "$PY" -m pip install -q pytest >/dev/null 2>&1 || true
fi

TEST=/tests/test_final_state.py
reward=0.0
if [ -f "$TEST" ]; then
    "$PY" -m pytest -q -rA "$TEST" > /tmp/test_output.txt 2>&1
    rc=$?
    log "pytest exit=$rc"
    tail -n 40 /tmp/test_output.txt 2>/dev/null || true
    [ "$rc" -eq 0 ] && reward=1.0
else
    log "missing $TEST"
fi

"$PY" - "$reward" <<'PY'
import json, os, sys
r = float(sys.argv[1])
tail = ""
if os.path.exists("/tmp/test_output.txt"):
    tail = open("/tmp/test_output.txt").read()[-1500:]
json.dump({"reward": r, "is_correct": r >= 1.0, "metadata": {"log_tail": tail}},
          open("/tmp/rllm/reward.json", "w"))
PY
exit 0
"""


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


def _materialize_task(task_dir: Path, row: dict, docker_image: str) -> None:
    """Expand a single raw TMax-15K row + image into a Harbor-format task tree."""
    task_dir.mkdir(parents=True, exist_ok=True)
    task_id = row["task_id"]

    (task_dir / "task.toml").write_text(
        _build_task_toml(task_id=task_id, domain=row.get("domain", ""), docker_image=docker_image),
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text((row.get("description") or "").strip() + "\n", encoding="utf-8")

    env_dir = task_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(_build_dockerfile(docker_image), encoding="utf-8")

    tests_dst = task_dir / "tests"
    tests_dst.mkdir(parents=True, exist_ok=True)
    (tests_dst / "test.sh").write_text(_VERIFIER_TEMPLATE, encoding="utf-8")
    (tests_dst / "test.sh").chmod(0o755)
    (tests_dst / "test_final_state.py").write_text(row.get("test_final_state") or "", encoding="utf-8")
    (tests_dst / "instance.json").write_text(
        json.dumps({"task_id": task_id, "domain": row.get("domain", "")}, indent=2),
        encoding="utf-8",
    )


def build_benchmark(
    *,
    name: str = "tmax-15k",
    split: str = "train",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    default_agent: str = "terminus-2",
    hf_repo_id: str | None = None,
    hf_split: str = "train",
    image_repo_id: str = IMAGE_HF_REPO_ID,
    clean: bool = False,
    register: bool = True,
) -> Path:
    """Materialize TMax-15K into a sandbox benchmark directory.

    Joins ``allenai/TMax-15K`` (verifier + description) with
    ``allenai/tmax-15k-open-instruct`` (prebuilt image) by ``task_id``. Tasks
    without a prebuilt image are skipped (the raw ``container_def`` is Apptainer
    format and not built here).
    """
    if catalog_entry:
        default_agent = catalog_entry.get("default_agent") or default_agent
        hf_repo_id = hf_repo_id or catalog_entry.get("source")
    hf_repo_id = hf_repo_id or DEFAULT_HF_REPO_ID

    out = Path(out_dir).expanduser()
    if clean and out.exists():
        logger.info("[tmax] removing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    image_map = _load_image_map(image_repo_id, hf_split)
    logger.info("[tmax] loading raw corpus %s split=%s ...", hf_repo_id, hf_split)
    rows = _load_rows(hf_repo_id, hf_split)

    if task_ids is not None:
        keep = set(task_ids)
        rows = [r for r in rows if r.get("task_id") in keep]
    if limit is not None:
        rows = rows[:limit]
    logger.info("[tmax] selected %d rows (task_ids=%s, limit=%s)", len(rows), task_ids and len(task_ids), limit)

    written = 0
    skipped_no_image = 0
    skipped_no_verifier = 0
    reg_rows: list[dict] = []
    for row in rows:
        task_id = row.get("task_id")
        if not task_id:
            continue
        docker_image = image_map.get(task_id)
        if not docker_image:
            skipped_no_image += 1
            continue
        if not (row.get("test_final_state") or "").strip():
            skipped_no_verifier += 1
            continue
        task_dst = out / task_id
        if task_dst.exists():
            shutil.rmtree(task_dst)
        _materialize_task(task_dst, row, docker_image)
        written += 1
        reg_rows.append(
            {
                "id": task_id,
                "task_id": task_id,
                "instruction": (row.get("description") or "").strip() + "\n",
                "task_path": str(task_dst),
                "docker_image": docker_image,
                "domain": row.get("domain", ""),
            }
        )

    description = (catalog_entry or {}).get("description") or ("TMax-15K (Ai2): compositional terminal-agent RL tasks with per-instance prebuilt Docker images and pytest test_final_state verifiers.")
    _write_dataset_toml(out, name=name, split=split, description=description, default_agent=default_agent)
    logger.info(
        "[tmax] wrote %d task dirs to %s (skipped: no image=%d, no verifier=%d)",
        written,
        out,
        skipped_no_image,
        skipped_no_verifier,
    )

    if not reg_rows:
        raise RuntimeError(
            f"[tmax] no tasks materialized from {hf_repo_id} (joined with {image_repo_id}). Skipped {skipped_no_image} without a prebuilt image, {skipped_no_verifier} without a verifier."
        )

    if register:
        try:
            from rllm.data import DatasetRegistry

            DatasetRegistry.register_dataset(
                name=name,
                data=reg_rows,
                split=split,
                source=hf_repo_id,
                description=description,
                category=(catalog_entry or {}).get("category", "agentic"),
            )
        except Exception:  # noqa: BLE001
            logger.warning("[tmax] could not register rows in DatasetRegistry (non-fatal)", exc_info=True)

    return out


def main() -> None:
    """CLI: ``python -m rllm.data.tmax_builder --out-dir <dir> [--limit N]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Materialize TMax-15K into an rLLM sandbox benchmark directory.")
    parser.add_argument("--out-dir", required=True, help="Output benchmark directory.")
    parser.add_argument("--name", default="tmax-15k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf-repo-id", default=None, help="Override raw HF source (default: allenai/TMax-15K).")
    parser.add_argument("--image-repo-id", default=IMAGE_HF_REPO_ID)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--default-agent", default="terminus-2")
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
        hf_repo_id=args.hf_repo_id,
        image_repo_id=args.image_repo_id,
        hf_split=args.hf_split,
        clean=args.clean,
        register=False,
    )


if __name__ == "__main__":
    main()

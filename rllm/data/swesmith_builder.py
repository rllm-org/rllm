"""Builder for the rllm-swesmith sandbox benchmark.

``kylemontgomery/swesmith-filtered`` is a flat HF dataset repo of ~4.8K
harbor-format task directories (105 Python repos), each shipping
``task.toml``, ``instruction.md``, ``environment/Dockerfile``,
``solution/solve.sh``, and ``tests/`` (an in-sandbox pytest verifier that
writes ``/logs/verifier/reward.txt``). That layout is already rLLM's
``type="sandbox"`` task-per-directory shape, so building the benchmark is:
download, screen out unsolvable instances, patch resource limits, and write
a ``dataset.toml``.

Single source of truth for materializing the dataset, used by
``rllm dataset pull rllm-swesmith`` (via the ``builder`` field in
``rllm/registry/datasets.json`` → :func:`rllm.cli._pull.pull_dataset`).

On-disk output (``<out_dir>/``):

    rllm-swesmith/
    ├── dataset.toml             # [dataset] type="sandbox", default_agent="mini-swe-agent"
    ├── <instance_id>/           # the harbor task dir, verbatim + patched task.toml
    └── ...

Registered rows carry ``task_path`` pointing at each task dir, so the
name-based flows (``rllm eval rllm-swesmith``, ``rllm train rllm-swesmith``)
root every Task at the real on-disk directory.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ID = "kylemontgomery/swesmith-filtered"

# Headroom over ``limit`` when selecting task dirs to download, covering
# instances the solvability screen will drop (~2% measured; see
# :func:`bug_in_test_file`).
_SCREEN_MARGIN = 1.2

# Upstream tasks declare no [environment] resources; remote backends then run
# at their defaults (Daytona: 1 GiB), which OOMs the verifier's in-sandbox
# ``uv add pytest swebench datasets swesmith`` resolution. Docker ignores these.
_DEFAULT_RESOURCES = {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240}


def bug_in_test_file(task_dir: Path) -> bool:
    """True when the injected bug lives in a FAIL_TO_PASS test file.

    Each instance branch is ``[base] → Bug Patch → Remove F2P Tests``: the
    agent works at the tip, where the F2P test files are deleted, and the
    verifier's ``git checkout HEAD~1`` restores them around the agent's
    uncommitted fix. When the bug patch itself targets a F2P test file
    (e.g. gpxpy keeps its whole suite in root ``test.py``), the deleted file
    *is* the one needing the fix — the task cannot be solved, and even the
    oracle's ``git apply --reverse`` fails on it.
    """
    cfg = json.loads((task_dir / "tests" / "config.json").read_text())
    targets = [line.split(" b/")[-1] for line in cfg["patch"].splitlines() if line.startswith("diff --git")]
    f2p_files = {t.split("::")[0] for t in cfg["FAIL_TO_PASS"]}
    return any(t in f2p_files for t in targets)


def list_task_ids() -> list[str]:
    """List every task-dir name in the HF repo from one tree listing.

    A repo-tree listing is a handful of paginated API calls — unlike the
    per-file "resolver" requests of a download, it does not eat into the
    free-tier 5000-per-5-minutes quota.
    """
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    return sorted({f.split("/", 1)[0] for f in files if "/" in f})


def spread_task_ids(ids: list[str], n: int) -> list[str]:
    """Pick ``n`` ids round-robin across repos (id prefix before the first dot).

    A plain ``sorted(ids)[:n]`` would land entirely on the alphabetically
    first repo or two; training batches should see the dataset's diversity.
    """
    by_repo: dict[str, list[str]] = {}
    for t in sorted(ids):
        by_repo.setdefault(t.split(".")[0], []).append(t)
    picked: list[str] = []
    queues = sorted(by_repo.values(), key=lambda q: q[0])
    while len(picked) < n and queues:
        queues = [q for q in queues if q]
        for q in queues:
            if len(picked) >= n:
                break
            picked.append(q.pop(0))
    return sorted(picked)


def _snapshot_download_with_retry(patterns: list[str] | None) -> Path:
    """``snapshot_download`` that sleeps out HF 429 rate-limit windows.

    The free tier allows 5000 resolver requests per 5 minutes and every file
    is one request, so a full-repo pull (~33K files) is guaranteed to hit the
    limit several times. Progress is kept in the HF cache, so each retry
    resumes where the previous attempt stopped instead of starting over.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    while True:
        try:
            return Path(snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=patterns))
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status != 429:
                raise
            wait = 310.0
            retry_after = (getattr(e, "response", None) is not None and e.response.headers.get("Retry-After")) or None
            if retry_after:
                try:
                    wait = float(retry_after) + 10.0
                except ValueError:
                    pass
            print(f"[swesmith] HF rate limit hit (429) — sleeping {wait:.0f}s, then resuming from cache ...", flush=True)
            time.sleep(wait)


def patch_task_toml(task_dir: Path, resources: dict | None = None) -> None:
    """Append a default ``[environment]`` resource block when the task has none."""
    toml_path = task_dir / "task.toml"
    text = toml_path.read_text()
    if "[environment]" in text:
        return
    res = resources or _DEFAULT_RESOURCES
    block = "\n[environment]\n" + "".join(f"{k} = {v}\n" for k, v in res.items())
    toml_path.write_text(text + block)


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


def build_benchmark(
    *,
    name: str = "rllm-swesmith",
    split: str = "train",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    default_agent: str = "mini-swe-agent",
    clean: bool = False,
    register: bool = True,
) -> Path:
    """Materialize swesmith-filtered into a sandbox benchmark directory.

    Args:
        name: Dataset/registry name (also the dataset.toml ``name``).
        split: Split label written into dataset.toml and the registry.
        out_dir: Output benchmark directory.
        catalog_entry: Optional catalog entry (datasets.json); ``description``/
            ``default_agent``/``limit`` are read from it when present.
        task_ids: Build only these instance ids (downloads just their files).
            Default: derived from ``limit`` when given, else the full repo.
        limit: Keep only N solvable tasks. Without explicit ``task_ids`` the
            download itself is capped to ``limit`` (+screen margin) ids picked
            round-robin across repos — not first-N-alphabetical.
        default_agent: ``default_agent`` written into dataset.toml.
        clean: Remove ``out_dir`` before building.
        register: Also register ``task_path`` rows in ``DatasetRegistry`` so
            the name-based eval/train flows and ``rllm dataset list`` work.

    Returns:
        Path to the built benchmark directory.
    """
    if catalog_entry:
        default_agent = catalog_entry.get("default_agent") or default_agent
        limit = limit if limit is not None else catalog_entry.get("limit")

    out = Path(out_dir).expanduser()
    if clean and out.exists():
        logger.info("[swesmith] removing existing %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # With a limit but no explicit ids, select the download set up front from
    # one repo-tree listing so a capped build never fetches all ~33K files.
    if task_ids is None and limit is not None:
        task_ids = spread_task_ids(list_task_ids(), int(limit * _SCREEN_MARGIN) + 10)
        print(f"[swesmith] selected {len(task_ids)} candidate tasks across repos for limit={limit}", flush=True)

    patterns = [f"{t}/*" for t in task_ids] if task_ids else None
    if task_ids:
        print(f"[swesmith] downloading {len(task_ids)} task dirs (~{7 * len(task_ids)} files) from {REPO_ID} ...", flush=True)
    else:
        print(
            f"[swesmith] downloading the FULL {REPO_ID} repo (~4.8K tasks, ~33K files).\n[swesmith] HF free-tier rate limits make this take ~35-40 min; progress resumes from cache if interrupted.",
            flush=True,
        )
    cache = _snapshot_download_with_retry(patterns)

    candidates = sorted(d for d in cache.iterdir() if d.is_dir() and (d / "task.toml").exists())
    if task_ids:
        wanted = set(task_ids)
        candidates = [d for d in candidates if d.name in wanted]

    kept: list[Path] = []
    skipped = 0
    for d in candidates:
        if bug_in_test_file(d):
            skipped += 1
            continue
        kept.append(d)
        if limit is not None and len(kept) >= limit:
            break
    logger.info("[swesmith] %d tasks kept, %d skipped (bug injected into a F2P test file — unsolvable)", len(kept), skipped)

    for d in kept:
        dst = out / d.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(d, dst)
        patch_task_toml(dst)

    description = (catalog_entry or {}).get("description") or "SWE-smith filtered: bug-fixing tasks across 105 Python repos (in-sandbox pytest grading)."
    _write_dataset_toml(out, name=name, split=split, description=description, default_agent=default_agent)
    logger.info("[swesmith] wrote %d task dirs to %s", len(kept), out)

    # task_path rows root each Task at its real directory, so the bare-name
    # eval/train flows pick up the per-task Dockerfile, verifier, and timeouts.
    if register:
        try:
            from rllm.data import DatasetRegistry

            reg_rows = [
                {
                    "id": d.name,
                    "instruction": (out / d.name / "instruction.md").read_text(encoding="utf-8"),
                    "task_path": str(out / d.name),
                }
                for d in kept
            ]
            DatasetRegistry.register_dataset(
                name=name,
                data=reg_rows,
                split=split,
                source=REPO_ID,
                description=description,
                category=(catalog_entry or {}).get("category", "agentic"),
            )
        except Exception:  # registry parity is best-effort; the benchmark dir is what eval uses
            logger.warning("[swesmith] could not register rows in DatasetRegistry (non-fatal)", exc_info=True)

    return out

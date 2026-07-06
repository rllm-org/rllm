"""Builder for Harbor task corpora stored as a tarball on the Hugging Face Hub.

The publish side (once, from the machine that has the tasks)::

    tar -C ~/mono -I 'zstd -T0' -cf tb_v2_tasks.tar.zst tb_v2
    hf upload <user>/tb-v2-tasks tb_v2_tasks.tar.zst --repo-type dataset --private
    hf upload <user>/tb-v2-tasks subsets/subset8.json --repo-type dataset

The consume side (any machine, via ``rllm dataset pull <name>``): this builder
downloads the tarball with ``hf_hub_download`` (HF-cached), extracts it once
under ``~/.rllm/datasets/_hf_tasks/<repo>/`` (shared across all dataset
versions built from the same repo), selects a task subset, and registers the
rows. Dataset *versions* are catalog entries that differ only in
``builder_kwargs`` — either an inline ``task_ids`` list or a ``subset_file``
(a JSON list of task names stored in the same HF repo).

Catalog entry shape::

    "tb-v2-subset8": {
      "source": "ThWu/tb-v2-tasks",
      "builder": "rllm.data.hf_tasks_builder:build_benchmark",
      "builder_kwargs": {"archive": "tb_v2_tasks.tar.zst",
                          "subset_file": "subsets/subset8.json"},
      ...
    }

Requires being logged in to HF (``hf auth login``) for private repos.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_once(archive_path: Path, dest: Path) -> Path:
    """Extract the tarball into ``dest`` (idempotent via a marker file).

    Returns the directory whose immediate children are Harbor task dirs
    (unwrapping a single top-level wrapper dir if the archive ships one).
    """
    marker = dest / ".extracted"
    if not marker.exists():
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("[hf-tasks] extracting %s -> %s (one-time)...", archive_path, dest)
        subprocess.run(
            ["tar", "--use-compress-program=unzstd", "-xf", str(archive_path), "-C", str(dest)],
            check=True,
        )
        marker.touch()
    children = [d for d in dest.iterdir() if d.is_dir()]
    if len(children) == 1 and not (children[0] / "task.toml").exists():
        return children[0]
    return dest


def build_benchmark(
    *,
    name: str,
    split: str = "train",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    repo_id: str | None = None,
    archive: str = "tasks.tar.zst",
    revision: str | None = None,
    task_ids: list[str] | None = None,
    subset_file: str | None = None,
    limit: int | None = None,
    default_agent: str = "terminus-2",
    register: bool = True,
) -> Path:
    """Download an HF-hosted Harbor task tarball and register a subset of it."""
    from huggingface_hub import hf_hub_download

    from rllm import paths
    from rllm.data.local_tasks_builder import build_benchmark as build_local

    if catalog_entry:
        repo_id = repo_id or catalog_entry.get("source")
    if not repo_id:
        raise ValueError("repo_id is required (or set 'source' in the catalog entry)")

    archive_path = Path(hf_hub_download(repo_id=repo_id, filename=archive, repo_type="dataset", revision=revision))

    # One shared extraction per repo — every dataset version reuses it.
    cache_dir = Path(paths.datasets_dir()) / "_hf_tasks" / repo_id.replace("/", "__")
    tasks_root = _extract_once(archive_path, cache_dir)

    if subset_file is not None:
        if task_ids is not None:
            raise ValueError("Pass task_ids OR subset_file, not both")
        subset_path = Path(hf_hub_download(repo_id=repo_id, filename=subset_file, repo_type="dataset", revision=revision))
        task_ids = json.loads(subset_path.read_text(encoding="utf-8"))

    # Selection + symlinking + dataset.toml + registration are shared with the
    # local-directory builder; only the sourcing differs.
    return build_local(
        name=name,
        split=split,
        out_dir=out_dir,
        catalog_entry=catalog_entry,
        tasks_root=str(tasks_root),
        task_ids=task_ids,
        limit=limit,
        default_agent=default_agent,
        register=register,
    )

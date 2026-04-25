"""Harbor dataset loader: converts Harbor datasets/tasks into rLLM Dataset rows.

Bridges Harbor's directory-per-task format into rLLM's flat tabular format.
Each Harbor task directory becomes a dict row with ``task_path``, ``instruction``,
and metadata fields that downstream components (HarborRuntime, HarborEvaluator)
consume.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def harbor_task_to_row(task_dir: Path, task_name: str | None = None, task_digest: str | None = None) -> dict | None:
    """Convert a single Harbor task directory into a flat dict row.

    Args:
        task_dir: Absolute path to a Harbor task directory (must contain task.toml).
        task_name: Optional org/name identifier (e.g., "harbor/hello-world").
        task_digest: Optional content hash digest (e.g., "sha256:...").

    Returns:
        A dict suitable for rLLM Dataset consumption, or None if the task is invalid.
    """
    from harbor.models.task.task import Task

    task_dir = Path(task_dir).resolve()
    try:
        task = Task(task_dir)
    except Exception as e:
        logger.warning("Skipping invalid Harbor task at %s: %s", task_dir, e)
        return None

    # Skip multi-step tasks (if the installed Harbor version supports them)
    if getattr(task, "has_steps", False):
        logger.info("Skipping multi-step Harbor task: %s", task.name)
        return None

    # Also check config.steps for newer Harbor versions
    if hasattr(task.config, "steps") and task.config.steps:
        logger.info("Skipping multi-step Harbor task: %s", task.name)
        return None

    name = task_name or task.name
    category = ""
    if hasattr(task.config, "metadata") and task.config.metadata:
        category = getattr(task.config.metadata, "category", "") or ""

    return {
        "task_id": name,
        "task_path": str(task_dir),
        "instruction": task.instruction,
        "question": task.instruction,  # rLLM convention alias
        "ground_truth": None,  # Harbor uses container-based verification
        "harbor_task_name": name,
        "harbor_task_digest": task_digest or "",
        "harbor_category": category,
        "data_source": f"harbor:{name}",
    }


def load_harbor_dataset_from_local(dataset_dir: str | Path) -> list[dict]:
    """Load a Harbor dataset from a local directory containing dataset.toml.

    Parses the manifest, resolves task references to local paths, and converts
    each task to a flat row.

    Args:
        dataset_dir: Path to a directory containing ``dataset.toml``.

    Returns:
        List of task row dicts.
    """
    from harbor.models.dataset.manifest import DatasetManifest

    dataset_dir = Path(dataset_dir).resolve()
    manifest_path = dataset_dir / "dataset.toml"

    if not manifest_path.exists():
        raise FileNotFoundError(f"No dataset.toml found at {dataset_dir}")

    manifest = DatasetManifest.from_toml_file(manifest_path)

    rows: list[dict] = []
    for task_ref in manifest.get_unique_tasks():
        # For local datasets, tasks are expected to be subdirectories or
        # downloaded to the cache. Try to resolve the local path.
        task_id = task_ref.to_package_reference()
        try:
            task_path = task_id.get_local_path()
        except Exception:
            # Task not cached locally -- skip
            logger.warning("Task %s not found locally, skipping", task_ref.name)
            continue

        if not task_path.exists():
            logger.warning("Task directory %s does not exist, skipping", task_path)
            continue

        row = harbor_task_to_row(task_path, task_name=task_ref.name, task_digest=task_ref.digest)
        if row is not None:
            rows.append(row)

    logger.info("Loaded %d tasks from local Harbor dataset at %s", len(rows), dataset_dir)
    return rows


def load_harbor_dataset_from_registry(identifier: str) -> list[dict]:
    """Load a Harbor dataset from the Harbor package registry.

    Downloads all referenced tasks and converts them to flat rows.

    Args:
        identifier: Dataset identifier, e.g., "terminal-bench" or "terminal-bench@2.0".
            The ``harbor:`` prefix should already be stripped by the caller.

    Returns:
        List of task row dicts.
    """
    from harbor.registry.client.factory import RegistryClientFactory

    async def _download():
        client = RegistryClientFactory.create()
        items = await client.download_dataset(identifier)
        return items

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            items = pool.submit(lambda: asyncio.run(_download())).result()
    else:
        items = asyncio.run(_download())

    rows: list[dict] = []
    for item in items:
        task_path = item.downloaded_path
        task_name = item.id.get_name() if hasattr(item.id, "get_name") else str(item.id)
        digest = ""
        if hasattr(item.id, "ref") and item.id.ref:
            digest = item.id.ref

        row = harbor_task_to_row(task_path, task_name=task_name, task_digest=digest)
        if row is not None:
            rows.append(row)

    logger.info("Loaded %d tasks from Harbor registry dataset '%s'", len(rows), identifier)
    return rows


def load_harbor_dataset(identifier: str) -> list[dict]:
    """Load a Harbor dataset by identifier.

    Handles both local paths and registry references.

    Args:
        identifier: One of:
            - An absolute path to a directory containing ``dataset.toml``
            - A registry name like ``"terminal-bench"`` or ``"terminal-bench@2.0"``
            - A path to a directory containing individual task directories

    Returns:
        List of task row dicts.
    """
    path = Path(identifier)

    # Check if it's a local dataset directory
    if path.is_absolute() and path.exists():
        if (path / "dataset.toml").exists():
            return load_harbor_dataset_from_local(path)
        # Maybe it's a directory of task directories
        if (path / "task.toml").exists():
            # Single task
            row = harbor_task_to_row(path)
            return [row] if row else []
        # Scan subdirectories for tasks
        rows = []
        for child in sorted(path.iterdir()):
            if child.is_dir() and (child / "task.toml").exists():
                row = harbor_task_to_row(child)
                if row:
                    rows.append(row)
        if rows:
            return rows

    # Otherwise, treat as a registry identifier
    return load_harbor_dataset_from_registry(identifier)

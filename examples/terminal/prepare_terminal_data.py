from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from terminal_bench.dataset.dataset import Dataset

def load_terminal_bench_dataset(
    dataset_name: str,
    dataset_version: str = "head",
    task_ids: Optional[List[str]] = None,
    n_tasks: Optional[int] = None,
    cache_path: Optional[Path] = None,
    local_registry_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load Terminal-Bench dataset and convert to minimal rLLM task dicts.

    Args:
        dataset_name: Dataset registry name.
        dataset_version: Concrete version or "head".
        task_ids: Optional subset of task IDs to include.
        n_tasks: Optional cap on number of tasks.
        cache_path: Optional path for dataset cache.
        local_registry_path: Optional path to a local registry.

    Returns:
        List[Dict[str, Any]]: Each dict includes ``task_path``, ``task_id``,
        and ``instruction``.
    """
    dataset = Dataset(
        name=dataset_name,
        version=dataset_version,
        task_ids=task_ids,
        n_tasks=n_tasks,
        local_registry_path=local_registry_path,
    )

    tasks: List[Dict[str, Any]] = []
    for task_path in dataset:
        task_config = load_task_config(task_path)

        task_dict = {
            "task_path": str(task_path),
            "task_id": task_path.name,
            "instruction": task_config["instruction"],
        }
        tasks.append(task_dict)

    return tasks


def load_task_config(task_path: Path) -> Dict[str, Any]:
    """Load and validate task configuration from task.yaml file.

    Args:
        task_path: Path to a Terminal-Bench task directory.

    Returns:
        Dict[str, Any]: Parsed YAML mapping.

    Raises:
        FileNotFoundError: If ``task.yaml`` is missing.
        ValueError: If required fields are missing.
    """
    task_yaml_path = task_path / "task.yaml"
    
    if not task_yaml_path.exists():
        raise FileNotFoundError(f"task.yaml not found at {task_yaml_path}")
    
    with open(task_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ["instruction"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in {task_yaml_path}")
    
    return config
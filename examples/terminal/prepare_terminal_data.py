"""
Prepare and register Terminal-Bench datasets for rLLM.

This mirrors the structure of examples/swe/prepare_swe_data.py but sources
tasks via the Terminal-Bench registry (no local registry path required).

Requires: pip install terminal-bench
"""

from __future__ import annotations

import json
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
    local_registry_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Load Terminal-Bench dataset and convert to rLLM task format.
    
    Args:
        dataset_name: Name of Terminal-Bench dataset
        dataset_version: Version of dataset to load
        task_ids: Specific task IDs to load
        n_tasks: Maximum number of tasks to load
        cache_path: Custom cache directory
        local_registry_path: Optional local registry.json path to use instead of remote
        
    Returns:
        List of task dictionaries for rLLM consumption
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
            "parser": task_config.get("parser", "pytest"),
            "max_agent_timeout_sec": task_config.get("max_agent_timeout_sec", 1800),
            "max_test_timeout_sec": task_config.get("max_test_timeout_sec", 120),
            "disable_asciinema": task_config.get("disable_asciinema", False),
            "run_tests_in_same_shell": task_config.get("run_tests_in_same_shell", False),
        }
        tasks.append(task_dict)

    return tasks


def load_task_config(task_path: Path) -> Dict[str, Any]:
    """Load and validate task configuration from task.yaml file."""
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

if __name__ == "__main__":
    tasks = load_terminal_bench_dataset(
        dataset_name="terminal-bench-core",
        dataset_version="0.1.1",  # or "head" for latest
    )

    print("\nSummary:")
    print(f"Num tasks: {len(tasks)}")
    if len(tasks) > 0:
        print("Sample entry:")
        print(json.dumps(tasks[0], indent=2))



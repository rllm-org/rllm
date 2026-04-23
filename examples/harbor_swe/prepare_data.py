from pathlib import Path

from datasets import Dataset

from rllm.data.dataset import DatasetRegistry

SWESMITH_TASK_ROOT = Path("external/harbor/datasets/swesmith").resolve()
SWEBENCH_TASK_ROOT = Path("external/harbor/datasets/swebench_verified").resolve()


def _load_harbor_tasks(task_root: Path) -> list[dict]:
    """Load all harbor task directories that have a task.toml and non-empty instruction.md."""
    task_dirs = sorted(p for p in task_root.iterdir() if p.is_dir() and (p / "task.toml").exists() and (p / "instruction.md").exists() and (p / "instruction.md").stat().st_size > 1)
    return [
        {
            "id": p.name,
            "task_path": str(p),
            "data_source": task_root.name,
        }
        for p in task_dirs
    ]


def prepare_swesmith_data():
    rows = _load_harbor_tasks(SWESMITH_TASK_ROOT)
    ds = Dataset.from_list(rows)
    ds = DatasetRegistry.register_dataset("swesmith_harbor", ds, "train")
    print(f"[swesmith] Registered {len(ds)} tasks")
    return ds


def prepare_swebench_verified_data():
    rows = _load_harbor_tasks(SWEBENCH_TASK_ROOT)
    ds = Dataset.from_list(rows)
    ds = DatasetRegistry.register_dataset("swebench_verified_harbor", ds, "test")
    print(f"[swebench_verified] Registered {len(ds)} tasks")
    return ds


if __name__ == "__main__":
    prepare_swesmith_data()
    prepare_swebench_verified_data()

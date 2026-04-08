from pathlib import Path

from datasets import Dataset

from rllm.data.dataset import DatasetRegistry

TASK_ROOT = Path("external/harbor/datasets/swesmith").resolve()


def prepare_swesmith_data():
    task_dirs = sorted(p for p in TASK_ROOT.iterdir() if p.is_dir() and (p / "task.toml").exists() and (p / "instruction.md").exists() and (p / "instruction.md").stat().st_size > 1)
    rows = [
        {
            "id": p.name,
            "task_path": str(p),
            "data_source": "swesmith",
        }
        for i, p in enumerate(task_dirs)
    ]
    ds = Dataset.from_list(rows)
    train_ds = DatasetRegistry.register_dataset("swesmith_harbor", ds, "train")
    print(f"Registered {len(train_ds)} tasks")
    print(f"Sample: {train_ds[0]}")
    return train_ds


if __name__ == "__main__":
    prepare_swesmith_data()

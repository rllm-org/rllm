"""Generate and register the ALFWorld train/test datasets.

Each row contains metadata for a single ALFWorld game: game_file path,
task_type, task_id, uid, and a human-readable question field.
The actual game environment is initialized from the game_file inside
the AgentFlow.

Usage::

    python tbmf/alfworld/prepare_alfworld_data.py
    python tbmf/alfworld/prepare_alfworld_data.py --data-root ./datasets/alfworld/json_2.1.1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rllm.data.dataset import DatasetRegistry

# Task type ID to human-readable name
TASK_TYPES = {
    "pick_and_place_simple": 1,
    "look_at_obj_in_light": 2,
    "pick_clean_then_place_in_recep": 3,
    "pick_heat_then_place_in_recep": 4,
    "pick_cool_then_place_in_recep": 5,
    "pick_two_obj_and_place": 6,
}

TASK_TYPE_NAMES = {v: k for k, v in TASK_TYPES.items()}


def _detect_task_type(game_path: str) -> str:
    """Detect task type from the game file path."""
    for task_name in TASK_TYPES:
        if task_name in game_path:
            return task_name
    return "unknown"


def _collect_game_files(data_dir: str) -> list[str]:
    """Recursively find all game.tw-pddl files under data_dir."""
    game_files = []
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f == "game.tw-pddl":
                game_files.append(os.path.join(root, f))
    game_files.sort()
    return game_files


def _extract_task_id(game_file: str) -> str:
    """Extract task/trial ID from the game file path."""
    parts = Path(game_file).parts
    for part in reversed(parts):
        if part.startswith("trial_"):
            return part
    # Fallback: use parent directory name
    return Path(game_file).parent.name


def _build_question(task_type: str, game_file: str) -> str:
    """Build a human-readable question string for dataset previews."""
    task_dir = Path(game_file).parent.parent.name
    return f"ALFWorld task: {task_type} ({task_dir})"


def prepare_alfworld_data(
    data_root: str | None = None,
    max_steps: int = 50,
):
    """Scan game files and register train/test datasets.

    Args:
        data_root: Path to json_2.1.1 directory. Defaults to datasets/alfworld/json_2.1.1
                   relative to repo root.
        max_steps: Maximum episode steps for each task.
    """
    if data_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        data_root = str(repo_root / "datasets" / "alfworld" / "json_2.1.1")

    data_root = os.path.realpath(data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"ALFWorld data root not found: {data_root}\n"
            f"Ensure the json_2.1.1 symlink exists at datasets/alfworld/json_2.1.1"
        )

    # Collect game files by split
    splits = {
        "train": os.path.join(data_root, "train"),
        "eval_id": os.path.join(data_root, "valid_seen"),
        "eval_ood": os.path.join(data_root, "valid_unseen"),
    }

    all_rows: dict[str, list[dict]] = {}

    for split_name, split_dir in splits.items():
        if not os.path.isdir(split_dir):
            print(f"  Warning: split directory not found: {split_dir}, skipping")
            continue

        game_files = _collect_game_files(split_dir)
        rows = []
        for game_file in game_files:
            task_type = _detect_task_type(game_file)
            task_id = _extract_task_id(game_file)
            uid = task_id
            rows.append({
                "uid": uid,
                "game_file": game_file,
                "task_type": task_type,
                "task_id": task_id,
                "split": split_name,
                "max_steps": max_steps,
                "question": _build_question(task_type, game_file),
            })
        all_rows[split_name] = rows

    # Register datasets
    registered = {}

    if "train" in all_rows:
        train_ds = DatasetRegistry.register_dataset("alfworld", all_rows["train"], "train")
        registered["train"] = train_ds
        print(f"  Train: {len(all_rows['train'])} games")

    # Use eval_id as the default "test" split
    if "eval_id" in all_rows:
        test_ds = DatasetRegistry.register_dataset("alfworld", all_rows["eval_id"], "test")
        registered["test"] = test_ds
        print(f"  Test (eval_id/valid_seen): {len(all_rows['eval_id'])} games")

    if "eval_ood" in all_rows:
        ood_ds = DatasetRegistry.register_dataset("alfworld", all_rows["eval_ood"], "eval_ood")
        registered["eval_ood"] = ood_ds
        print(f"  Eval OOD (valid_unseen): {len(all_rows['eval_ood'])} games")

    return registered


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare ALFWorld datasets for rllm")
    ap.add_argument("--data-root", type=str, default=None,
                    help="Path to json_2.1.1 directory")
    ap.add_argument("--max-steps", type=int, default=50,
                    help="Max episode steps per task")
    args = ap.parse_args()

    print("Preparing ALFWorld datasets...")
    registered = prepare_alfworld_data(
        data_root=args.data_root,
        max_steps=args.max_steps,
    )

    if registered:
        # Print a sample
        for split_name, ds in registered.items():
            data = ds.get_data()
            if data:
                print(f"\n  Sample ({split_name}): {data[0]}")
    else:
        print("No datasets registered!")


if __name__ == "__main__":
    main()

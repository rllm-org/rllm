#!/usr/bin/env python3
"""Prepare SWE-bench data for Tinker training.

Registers datasets in rllm DatasetRegistry. The entire dataset is registered
as-is; train/val assignment happens in train_swe_tinker.py.

Usage:
    python prepare_tinker_data.py --dataset swe_smith
    python prepare_tinker_data.py --dataset swe_bench_multilingual --split test
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rllm.data.dataset import DatasetRegistry
from data import prepare_dataset, DATASET_CONFIGS


def serialize_complex_fields(data: list[dict]) -> list[dict]:
    """Serialize complex dict/list fields as JSON strings for parquet compatibility."""
    complex_fields = ["spec_dict", "fail_to_pass", "pass_to_pass", "FAIL_TO_PASS", "PASS_TO_PASS"]

    result = []
    for task in data:
        task_copy = dict(task)
        for field in complex_fields:
            if field in task_copy:
                value = task_copy[field]
                if isinstance(value, (dict, list)):
                    task_copy[field] = json.dumps(value)
        result.append(task_copy)
    return result


def prepare_swe_tinker_data(dataset_name: str, split: str = "train"):
    """Load and register a dataset in DatasetRegistry.

    Args:
        dataset_name: Dataset to prepare (key in DATASET_CONFIGS)
        split: HuggingFace split to load (train or test)
    """
    print(f"Loading {dataset_name} (split={split})...")
    data = prepare_dataset(dataset_name, split=split)
    print(f"Loaded {len(data)} instances")

    # Serialize complex fields for parquet compatibility
    data = serialize_complex_fields(data)

    DatasetRegistry.register_dataset(dataset_name, data)
    print(f"Registered '{dataset_name}' in DatasetRegistry")


def verify_registration(dataset_name: str):
    """Verify dataset is registered with required fields."""
    data = DatasetRegistry.load_dataset(dataset_name)
    if data is None:
        print(f"ERROR: {dataset_name} not found")
        return False

    print(f"OK: {dataset_name} ({len(data)} instances)")

    if len(data) > 0:
        required = ["instance_id", "problem_statement", "docker_image", "eval_type"]
        missing = [f for f in required if f not in data[0]]
        if missing:
            print(f"WARNING: missing fields: {missing}")
        else:
            print("OK: has all required fields")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SWE-bench data for Tinker training")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to prepare")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="HuggingFace split to load (default: train)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing registration")
    args = parser.parse_args()

    if args.verify_only:
        success = verify_registration(args.dataset)
        sys.exit(0 if success else 1)

    prepare_swe_tinker_data(args.dataset, args.split)
    verify_registration(args.dataset)

#!/usr/bin/env python3
"""Prepare a small easy mix of SWE-smith trajectory datasets for training.

Loads multiple HuggingFace datasets that have a `success_rate` column,
filters to instances where success_rate == 1.0, samples --per-language
from each, combines them, and registers in rllm DatasetRegistry.

Usage:
    python -m swe.scripts.prepare_easy_mix
    python -m swe.scripts.prepare_easy_mix --per-language 2 --seed 42
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_dataset
from rllm.data.dataset import DatasetRegistry

from swe.scripts.prepare_filtered_mix import (
    SOURCE_DATASETS,
    instance_to_task,
    serialize_complex_fields,
)


def prepare_easy_mix(
    per_language: int = 2,
    seed: int = 42,
    dataset_name: str = "swe_smith_easy",
) -> list[dict]:
    """Load, filter to success_rate == 1.0, sample, and combine datasets."""
    random.seed(seed)
    all_tasks = []

    for hf_id, lang in SOURCE_DATASETS:
        print(f"\n--- Loading {hf_id} ({lang}) ---")
        ds = load_dataset(hf_id, split="train")
        print(f"  Total instances: {len(ds)}")

        rates = ds["success_rate"]
        easy_indices = [i for i, r in enumerate(rates) if r == 1.0]
        print(f"  With success_rate == 1.0: {len(easy_indices)}")

        if len(easy_indices) < per_language:
            raise RuntimeError(
                f"{hf_id}: only {len(easy_indices)} easy instances, need {per_language}"
            )

        sampled_indices = random.sample(easy_indices, per_language)
        print(f"  Sampled {per_language} easy instances")

        sampled = ds.select(sampled_indices)
        tasks = [instance_to_task(row, hf_id) for row in sampled]
        all_tasks.extend(tasks)

    print(f"\n=== Combined total: {len(all_tasks)} instances ===")

    serialized = serialize_complex_fields(all_tasks)
    DatasetRegistry.register_dataset(dataset_name, serialized)
    print(f"Registered '{dataset_name}' in DatasetRegistry")

    loaded = DatasetRegistry.load_dataset(dataset_name)
    if loaded is not None:
        print(f"Verified: {len(loaded)} instances in registry")
        required = ["instance_id", "problem_statement", "docker_image", "eval_type"]
        missing = [f for f in required if f not in loaded[0]]
        if missing:
            print(f"WARNING: missing fields: {missing}")
        else:
            print("OK: has all required fields")
    else:
        print("ERROR: dataset not found after registration")

    return all_tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare easy mix of SWE-smith trajectory datasets")
    parser.add_argument("--per-language", type=int, default=2,
                        help="Instances per language (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--name", default="swe_smith_easy",
                        help="Dataset name for DatasetRegistry (default: swe_smith_easy)")
    args = parser.parse_args()

    prepare_easy_mix(
        per_language=args.per_language,
        seed=args.seed,
        dataset_name=args.name,
    )

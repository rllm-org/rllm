#!/usr/bin/env python3
"""Prepare a filtered mix of SWE-smith trajectory datasets for training.

Loads multiple HuggingFace datasets that have a `success_rate` column,
filters to instances where 0 < success_rate < 1, samples at most
--max-per-dataset from each, combines them, and registers in rllm DatasetRegistry.

Usage:
    python -m swe.scripts.prepare_filtered_mix
    python -m swe.scripts.prepare_filtered_mix --max-per-dataset 300 --seed 42
    python -m swe.scripts.prepare_filtered_mix --name my_custom_mix
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_dataset
from rllm.data.dataset import DatasetRegistry

# Source datasets: (hf_id, language_label)
SOURCE_DATASETS = [
    ("JWei05/swe_smith_py_qwen3.5_35b_trajs_1952", "python"),
    ("JWei05/swe_smith_rs_qwen3.5_35b_trajs_2477", "rust"),
    ("JWei05/swe_smith_go_qwen3.5_35b_trajs_1448", "go"),
    ("JWei05/swe_smith_js_qwen3.5_35b_trajs_4358", "javascript"),
    ("JWei05/swe_smith_java_qwen3.5_35b_trajs_4369", "java"),
]

WORKING_DIR = "/testbed"
EVAL_TYPE = "swesmith"


def get_docker_image(instance: dict) -> str:
    """Get Docker image URI for a SWE-smith instance."""
    if "image_name" in instance and instance["image_name"]:
        return instance["image_name"]
    repo = instance.get("repo", instance["instance_id"].rsplit(".", 1)[0])
    return f"swebench/swesmith.x86_64.{repo}".lower()


def instance_to_task(instance: dict, source_name: str) -> dict:
    """Convert a HuggingFace dataset row to a task dict."""
    get = lambda name, default="": instance.get(name, default) or default

    docker_image = get_docker_image(instance)

    f2p = instance.get("FAIL_TO_PASS", [])
    p2p = instance.get("PASS_TO_PASS", [])
    if not isinstance(f2p, list):
        f2p = []
    if not isinstance(p2p, list):
        p2p = []

    return {
        "instance_id": instance["instance_id"],
        "problem_statement": instance["problem_statement"],
        "repo": get("repo"),
        "base_commit": get("base_commit"),
        "version": get("version"),
        "test_patch": get("test_patch"),
        "patch": get("patch"),
        "docker_image": docker_image,
        "image_name": docker_image,
        "repo_name": get("repo"),
        "working_dir": WORKING_DIR,
        "eval_type": EVAL_TYPE,
        "data_source": source_name,
        "FAIL_TO_PASS": f2p,
        "PASS_TO_PASS": p2p,
        "fail_to_pass": f2p,
        "pass_to_pass": p2p,
    }


def serialize_complex_fields(data: list[dict]) -> list[dict]:
    """Serialize complex dict/list fields as JSON strings for parquet compatibility."""
    complex_fields = ["spec_dict", "fail_to_pass", "pass_to_pass", "FAIL_TO_PASS", "PASS_TO_PASS"]
    result = []
    for task in data:
        task_copy = dict(task)
        for field in complex_fields:
            if field in task_copy and isinstance(task_copy[field], (dict, list)):
                task_copy[field] = json.dumps(task_copy[field])
        result.append(task_copy)
    return result


def prepare_filtered_mix(
    max_per_dataset: int = 300,
    seed: int = 42,
    dataset_name: str = "swe_smith_filtered_mix",
    hard: bool = False,
) -> list[dict]:
    """Load, filter, sample, and combine datasets.

    Args:
        max_per_dataset: Maximum instances to sample from each source dataset.
        seed: Random seed for reproducibility.
        dataset_name: Name to register in DatasetRegistry.
        hard: If True, take the instances with lowest success_rate instead of random sampling.

    Returns:
        Combined list of task dicts.
    """
    random.seed(seed)
    all_tasks = []

    for hf_id, lang in SOURCE_DATASETS:
        print(f"\n--- Loading {hf_id} ({lang}) ---")
        ds = load_dataset(hf_id, split="train")
        print(f"  Total instances: {len(ds)}")

        # Filter using arrow column access (fast) instead of row-by-row iteration
        rates = ds["success_rate"]
        valid_indices = [i for i, r in enumerate(rates) if 0.0 < r < 1.0]
        print(f"  After filtering (0 < success_rate < 1): {len(valid_indices)}")

        # Select indices: hardest (lowest success_rate) or random
        if len(valid_indices) > max_per_dataset:
            if hard:
                # Sort by success_rate ascending, take the hardest (lowest non-zero)
                sorted_indices = sorted(valid_indices, key=lambda i: rates[i])
                sampled_indices = sorted_indices[:max_per_dataset]
                sr_min = rates[sampled_indices[0]]
                sr_max = rates[sampled_indices[-1]]
                print(f"  Took {max_per_dataset} hardest (success_rate {sr_min:.3f}–{sr_max:.3f})")
            else:
                sampled_indices = random.sample(valid_indices, max_per_dataset)
                print(f"  Sampled {max_per_dataset} instances (random)")
        else:
            sampled_indices = valid_indices
            print(f"  Kept all {len(sampled_indices)} instances (under limit of {max_per_dataset})")

        sampled = ds.select(sampled_indices)

        # Convert to task dicts
        tasks = [instance_to_task(row, hf_id) for row in sampled]
        all_tasks.extend(tasks)

    print(f"\n=== Combined total: {len(all_tasks)} instances ===")

    # Serialize and register
    serialized = serialize_complex_fields(all_tasks)
    DatasetRegistry.register_dataset(dataset_name, serialized)
    print(f"Registered '{dataset_name}' in DatasetRegistry")

    # Verify
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
    parser = argparse.ArgumentParser(description="Prepare filtered mix of SWE-smith trajectory datasets")
    parser.add_argument("--max-per-dataset", type=int, default=300,
                        help="Max instances to sample from each source dataset (default: 300)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--name", default="swe_smith_filtered_mix",
                        help="Dataset name for DatasetRegistry (default: swe_smith_filtered_mix)")
    parser.add_argument("--hard", action="store_true",
                        help="Take hardest instances (lowest success_rate) instead of random")
    args = parser.parse_args()

    name = args.name
    if args.hard and name == "swe_smith_filtered_mix":
        name = "swe_smith_filtered_mix_hard"

    prepare_filtered_mix(
        max_per_dataset=args.max_per_dataset,
        seed=args.seed,
        dataset_name=name,
        hard=args.hard,
    )

#!/usr/bin/env python3
"""Prepare SWE-bench data for the workflow.

Usage:
    python data.py --dataset swe_bench_pro --split test --output data/swe_bench_pro.json
    python data.py --dataset swe_bench_multilingual --split test --output data/swe_bench_multilingual.json
"""

import json
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset


# Path to SWE-bench Pro CSV (same as swe_bench_pro_eval.py uses)
SWE_BENCH_PRO_CSV = Path(__file__).parent / "SWE-bench_Pro-os" / "swe_bench_pro_full.csv"

# Add helper_code to path for image URI generation
_HELPER_CODE_PATH = Path(__file__).parent / "SWE-bench_Pro-os" / "helper_code"
sys.path.insert(0, str(_HELPER_CODE_PATH))
try:
    from image_uri import get_dockerhub_image_uri as _get_swebench_pro_image_uri
except ImportError:
    _get_swebench_pro_image_uri = None

DATASET_CONFIGS = {
    "swe_bench_pro": {
        "source": "csv",  # Load from CSV to align with swe_bench_pro_eval.py
        "csv_path": SWE_BENCH_PRO_CSV,
        "docker_prefix": "jefzda/sweap-images",
        "working_dir": "/app",
        "eval_type": "swebench_pro",
    },
    "swe_bench_multilingual": {
        "source": "swe-bench/SWE-Bench_Multilingual",
        "docker_prefix": "swebench/sweb.eval.x86_64",
        "working_dir": "/testbed",
        "eval_type": "swebench",
    },
    "swe_smith": {
        "source": "combined",
        "components": ["swe_smith_py", "swe_smith_go"],
        "working_dir": "/testbed",
        "eval_type": "swesmith",
    },
    "swe_smith_py": {
        "source": "JWei05/SWE-smith-py-39471-filtered-for-problem-statements",
        "working_dir": "/testbed",
        "eval_type": "swesmith",
        "sample_size": 2000,
        "sample_seed": 42,
    },
    "swe_smith_go": {
        "source": "JWei05/SWE-smith-go-1629-filtered-for-problem-statements",
        "working_dir": "/testbed",
        "eval_type": "swesmith",
        "sample_size": 1629,
        "sample_seed": 42,
    },
    "swe_smith_js": {
        "source": "SWE-bench/SWE-smith-js",
        "working_dir": "/testbed",
        "eval_type": "swesmith",
        "sample_size": 5000,
        "sample_seed": 42,
    },
    "swe_smith_rs": {
        "source": "SWE-bench/SWE-smith-rs",
        "working_dir": "/testbed",
        "eval_type": "swesmith",
        "sample_size": 5000,
        "sample_seed": 42,
    },
    "swe_smith_java": {
        "source": "JWei05/SWE-smith-java-6450-filtered",
        "working_dir": "/testbed",
        "eval_type": "swesmith",
        "sample_size": 5000,
        "sample_seed": 42,
    },
}


def get_docker_image_uri(instance: dict, config: dict, dockerhub_username: str = "jefzda") -> str:
    """Get Docker image URI based on dataset type.

    Args:
        instance: Instance dict with instance_id and repo fields
        config: Dataset config dict with eval_type and docker_prefix
        dockerhub_username: DockerHub username for SWE-bench Pro images

    Returns:
        Docker image URI string
    """
    instance_id = instance["instance_id"]
    eval_type = config["eval_type"]

    if eval_type == "swebench_pro":
        if _get_swebench_pro_image_uri:
            return _get_swebench_pro_image_uri(instance_id, dockerhub_username, instance.get("repo", ""))
        prefix = config["docker_prefix"]
        repo = instance.get("repo", "")
        if "/" in repo:
            repo_base, repo_name = repo.lower().split("/")
            return f"{prefix}:{repo_base}.{repo_name}-{instance_id}"
        return f"{prefix}:{instance_id}"

    if eval_type == "swesmith":
        # SWE-smith: use image_name from instance if available
        if "image_name" in instance and instance["image_name"]:
            return instance["image_name"]
        # Fallback: construct from repo field (owner__repo.commit)
        repo = instance.get("repo", instance_id.rsplit(".", 1)[0])
        return f"swebench/swesmith.x86_64.{repo}".lower()

    # Standard SWE-bench multilingual: swebench/sweb.eval.x86_64.{id_docker_compatible}:latest
    prefix = config["docker_prefix"]
    id_docker_compatible = instance_id.replace("__", "_1776_")
    return f"{prefix}.{id_docker_compatible}:latest".lower()


def prepare_dataset(
    dataset_name: str,
    split: str = "test",
    output_path: str = None,
    dockerhub_username: str = "jefzda",
    include_golden_patch: bool = False,
) -> list[dict]:
    """Load and prepare a SWE-bench dataset.

    Args:
        dataset_name: Name of dataset (key in DATASET_CONFIGS)
        split: Dataset split (test, dev, train)
        output_path: Optional path to save prepared data as JSON
        dockerhub_username: DockerHub username for Pro images
        include_golden_patch: If True, include ground-truth patch as "golden_patch" field

    Returns:
        List of task dicts ready for the workflow
    """
    import random

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(DATASET_CONFIGS.keys())}")

    if dataset_name.startswith("swe_smith") and split != "train":
        print(f"Overriding split to train for {dataset_name} because it's a SWE-smith dataset")
        split = "train"

    config = DATASET_CONFIGS[dataset_name]

    # Handle combined datasets (recursively load components)
    if config["source"] == "combined":
        all_tasks = []
        for component in config["components"]:
            print(f"Loading component: {component}")
            component_tasks = prepare_dataset(
                component, split, None, dockerhub_username,
                include_golden_patch=include_golden_patch,
            )
            all_tasks.extend(component_tasks)
        print(f"Combined {len(all_tasks)} total instances from {len(config['components'])} components")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(all_tasks, f, indent=2)
            print(f"Saved {len(all_tasks)} tasks to {output_path}")

        return all_tasks

    # Load dataset from appropriate source
    if config["source"] == "csv":
        csv_path = config["csv_path"]

        if not csv_path.exists():
            print(f"CSV not found at {csv_path}")
            print(f"Downloading SWE-bench Pro dataset from HuggingFace...")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            hf_dataset = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
            df = hf_dataset.to_pandas()
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV to {csv_path}")
            dataset = df.to_dict('records')
        else:
            print(f"Loading dataset from CSV: {csv_path}...")
            df = pd.read_csv(csv_path)
            dataset = df.to_dict('records')

        print(f"Loaded {len(dataset)} instances from CSV")
    else:
        print(f"Loading dataset {config['source']} split={split}...")
        dataset = list(load_dataset(config["source"], split=split))
        print(f"Loaded {len(dataset)} instances")

    # Sample if configured
    if "sample_size" in config and config["sample_size"] < len(dataset):
        seed = config.get("sample_seed", 42)
        random.seed(seed)
        dataset = random.sample(dataset, config["sample_size"])
        print(f"Sampled {len(dataset)} instances (seed={seed})")

    tasks = []
    eval_type = config["eval_type"]

    for instance in dataset:
        get_field = lambda name, default="": instance.get(name, default) or default

        # Core fields
        task = {
            "instance_id": instance["instance_id"],
            "problem_statement": instance["problem_statement"],
            "repo": get_field("repo"),
            "base_commit": get_field("base_commit"),
            "version": get_field("version"),
            "test_patch": get_field("test_patch"),
            "docker_image": get_docker_image_uri(instance, config, dockerhub_username),
            "image_name": get_docker_image_uri(instance, config, dockerhub_username),
            "repo_name": get_field("repo"),
            "working_dir": config["working_dir"],
            "eval_type": eval_type,
            "data_source": dataset_name,
        }

        # Handle test fields based on dataset type
        if eval_type == "swesmith":
            task["patch"] = get_field("patch")
            # SWE-smith uses uppercase list fields directly
            f2p = instance.get("FAIL_TO_PASS", [])
            p2p = instance.get("PASS_TO_PASS", [])
            task["FAIL_TO_PASS"] = f2p if isinstance(f2p, list) else []
            task["PASS_TO_PASS"] = p2p if isinstance(p2p, list) else []
            task["fail_to_pass"] = f2p
            task["pass_to_pass"] = p2p
        else:
            # SWE-bench Pro/Multilingual use JSON strings or lists
            f2p = instance.get("fail_to_pass") or instance.get("FAIL_TO_PASS", "[]")
            p2p = instance.get("pass_to_pass") or instance.get("PASS_TO_PASS", "[]")
            task["fail_to_pass"] = f2p
            task["pass_to_pass"] = p2p
            task["FAIL_TO_PASS"] = f2p
            task["PASS_TO_PASS"] = p2p

        # SWE-bench Pro specific fields
        if eval_type == "swebench_pro":
            task["before_repo_set_cmd"] = get_field("before_repo_set_cmd")
            task["selected_test_files_to_run"] = get_field("selected_test_files_to_run", "[]")

        if include_golden_patch:
            task["golden_patch"] = get_field("patch")

        tasks.append(task)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(tasks, f, indent=2)
        print(f"Saved {len(tasks)} tasks to {output_path}")

    return tasks


def load_prepared_data(path: str) -> list[dict]:
    """Load previously prepared data from JSON file."""
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare SWE-bench data for evaluation")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to prepare")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--dockerhub_username", default="jefzda",
                        help="DockerHub username for SWE-bench Pro images")
    args = parser.parse_args()

    tasks = prepare_dataset(
        args.dataset,
        args.split,
        args.output,
        args.dockerhub_username,
    )
    print(f"Prepared {len(tasks)} tasks from {args.dataset}")

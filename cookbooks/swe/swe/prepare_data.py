#!/usr/bin/env python3
"""Prepare SWE data for evaluation and training workflows.

Usage:
    python swe/prepare_data.py --dataset swe_bench_pro --split test --output data/swe_bench_pro.json
    python swe/prepare_data.py --dataset swe_bench_multilingual --split test --output data/swe_bench_multilingual.json
"""

import json
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset


# Add helper_code to path for image URI generation
HELPER_CODE_PATH = Path(__file__).parent.parent / "external" / "SWE-bench_Pro-os" / "helper_code"
sys.path.insert(0, str(HELPER_CODE_PATH))

try:
    from image_uri import get_dockerhub_image_uri as get_swebench_pro_image_uri
except ImportError:
    get_swebench_pro_image_uri = None


# Path to SWE-bench Pro CSV (same as swe_bench_pro_eval.py uses)
SWE_BENCH_PRO_CSV = Path(__file__).parent.parent / "external" / "SWE-bench_Pro-os" / "swe_bench_pro_full.csv"

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
    "swe_smith_ts": {
        "source": "SWE-bench/SWE-smith-ts",
        "working_dir": "/testbed",
        "eval_type": "swesmith",
        "sample_size": 5000,
        "sample_seed": 42,
    },
    "swe_rebench_v2": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
    },
    "swe_rebench_v2_py": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "python",
    },
    "swe_rebench_v2_ts": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "ts",
    },
    "swe_rebench_v2_js": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "js",
    },
    "swe_rebench_v2_go": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "go",
    },
    "swe_rebench_v2_rs": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "rust",
    },
    "swe_rebench_v2_java": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "java",
    },
    "swe_rebench_v2_c_cpp": {
        "source": "combined",
        "components": ["swe_rebench_v2_c", "swe_rebench_v2_cpp"],
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
    },
    "swe_rebench_v2_cpp": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "cpp",
    },
    "swe_rebench_v2_c": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "c",
    },
    "swe_rebench_v2_php": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "php",
    },
    "swe_rebench_v2_kotlin": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "kotlin",
    },
    "swe_rebench_v2_scala": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "scala",
    },
    "swe_rebench_v2_swift": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "swift",
    },
    "swe_rebench_v2_ruby": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "ruby",
    },
    "swe_rebench_v2_csharp": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "csharp",
    },
    "swe_rebench_v2_julia": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "julia",
    },
    "swe_rebench_v2_elixir": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "elixir",
    },
    "swe_rebench_v2_dart": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "dart",
    },
    "swe_rebench_v2_r": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "r",
    },
    "swe_rebench_v2_clojure": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "clojure",
    },
    "swe_rebench_v2_ocaml": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "ocaml",
    },
    "swe_rebench_v2_lua": {
        "source": "nebius/SWE-rebench-V2",
        "working_dir": "dynamic",
        "eval_type": "swe_rebench_v2",
        "language_filter": "lua",
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
        prefix = config["docker_prefix"]
        if get_swebench_pro_image_uri:
            return get_swebench_pro_image_uri(instance_id, dockerhub_username, instance.get("repo", ""))
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

    if eval_type == "swe_rebench_v2":
        if "image_name" in instance and instance["image_name"]:
            return instance["image_name"]
        return f"swerebenchv2/{instance_id}"

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

    if (dataset_name.startswith("swe_smith") or dataset_name.startswith("swe_rebench_v2")) and split != "train":
        print(f"Overriding split to train for {dataset_name} because it only has a train split")
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

    # Filter by language if configured (for swe_rebench_v2 per-language datasets)
    if "language_filter" in config:
        lang_filter = config["language_filter"]
        dataset = [inst for inst in dataset if inst.get("language", "") == lang_filter]
        print(f"Filtered to {len(dataset)} instances for language={lang_filter}")

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

        # Resolve working directory (dynamic for swe_rebench_v2)
        if config["working_dir"] == "dynamic":
            repo = instance.get("repo", "")
            working_dir = f"/{repo.split('/')[1]}" if "/" in repo else "/testbed"
        else:
            working_dir = config["working_dir"]

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
            "working_dir": working_dir,
            "eval_type": eval_type,
            "data_source": dataset_name,
        }

        # Handle test fields based on dataset type
        if eval_type in ("swesmith", "swe_rebench_v2"):
            task["patch"] = get_field("patch")
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

        if eval_type == "swe_rebench_v2":
            task["language"] = instance.get("language", "")
            install_config = instance.get("install_config", {})
            if isinstance(install_config, str):
                import json as _json
                install_config = _json.loads(install_config)
            task["install_config"] = install_config

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

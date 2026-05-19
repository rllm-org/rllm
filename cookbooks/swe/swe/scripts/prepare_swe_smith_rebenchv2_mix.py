#!/usr/bin/env python3
"""Prepare a SWE-smith + SWE-rebench V2 training mix.

The SWE-smith portion intentionally matches the default
``swe_smith_filtered_mix`` recipe: five trajectory datasets, filtered to
0 < success_rate < 1, with up to 300 random rows per source.

Usage:
    python -m swe.scripts.prepare_swe_smith_rebenchv2_mix
    python -m swe.scripts.prepare_swe_smith_rebenchv2_mix --seed 123
    python -m swe.scripts.prepare_swe_smith_rebenchv2_mix --c-cpp-repeat-count 3
    python -m swe.scripts.prepare_swe_smith_rebenchv2_mix --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry
from swe.prepare_data import prepare_dataset
from swe.scripts.prepare_filtered_mix import (
    SOURCE_DATASETS,
    instance_to_task,
    serialize_complex_fields,
)


@dataclass(frozen=True)
class LengthBucket:
    name: str
    label: str
    lower: float
    upper: float | None


BUCKET_0_1_5 = LengthBucket("log_0_0_1_5", "[0.0, 1.5]", 0.0, 1.5)
BUCKET_1_5_2_0 = LengthBucket("log_1_5_2_0", "[1.5, 2.0]", 1.5, 2.0)

ALL_BUCKETS = [
    BUCKET_0_1_5,
    BUCKET_1_5_2_0,
    LengthBucket("log_2_0_2_5", "[2.0, 2.5]", 2.0, 2.5),
    LengthBucket("log_2_5_inf", "[2.5, inf)", 2.5, None),
]

REBENCH_DATASET_BY_LANGUAGE = {
    "python": "swe_rebench_v2_py",
    "go": "swe_rebench_v2_go",
    "php": "swe_rebench_v2_php",
    "java": "swe_rebench_v2_java",
    "js": "swe_rebench_v2_js",
    "ts": "swe_rebench_v2_ts",
    "c": "swe_rebench_v2_c",
    "cpp": "swe_rebench_v2_cpp",
}

DEFAULT_C_CPP_REPEAT_COUNT = 3


def patch_line_count(patch: object) -> int:
    if patch is None:
        return 0
    if isinstance(patch, float) and math.isnan(patch):
        return 0
    text = str(patch)
    if not text:
        return 0
    return len(text.splitlines())


def log10_patch_length(lines: int) -> float:
    return math.log10(max(lines, 1))


def assign_bucket(log_length: float) -> LengthBucket:
    for bucket in ALL_BUCKETS:
        if log_length < bucket.lower:
            continue
        if bucket.upper is None or log_length <= bucket.upper:
            return bucket
    return ALL_BUCKETS[-1]


def stable_sample_seed(seed: int, *parts: str) -> int:
    value = ":".join([str(seed), *parts])
    digest = hashlib.sha256(value.encode()).hexdigest()
    return int(digest[:8], 16)


def sample_tasks(candidates: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    candidates = sorted(candidates, key=lambda item: item["instance_id"])
    if len(candidates) <= count:
        return list(candidates)
    selected = random.Random(seed).sample(candidates, count)
    return sorted(selected, key=lambda item: item["instance_id"])


def repeat_task_rows(tasks: list[dict[str, Any]], repeat_count: int) -> list[dict[str, Any]]:
    if repeat_count < 1:
        raise ValueError(f"repeat_count must be >= 1, got {repeat_count}")

    repeated = []
    for task in tasks:
        for repeat_idx in range(repeat_count):
            task_copy = dict(task)
            task_copy["mix_repeat_index"] = repeat_idx
            task_copy["mix_repeat_count"] = repeat_count
            repeated.append(task_copy)
    return repeated


def load_swe_smith_filtered_tasks(max_per_dataset: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the same SWE-smith rows used by prepare_filtered_mix.py."""
    rng = random.Random(seed)
    all_tasks: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for hf_id, language in SOURCE_DATASETS:
        print(f"\n--- Loading {hf_id} ({language}) ---")
        dataset = load_dataset(hf_id, split="train")
        print(f"  Total instances: {len(dataset)}")

        rates = dataset["success_rate"]
        valid_indices = [i for i, rate in enumerate(rates) if 0.0 < rate < 1.0]
        print(f"  After filtering (0 < success_rate < 1): {len(valid_indices)}")

        if len(valid_indices) > max_per_dataset:
            sampled_indices = rng.sample(valid_indices, max_per_dataset)
            print(f"  Sampled {max_per_dataset} instances")
        else:
            sampled_indices = valid_indices
            print(f"  Kept all {len(sampled_indices)} instances")

        sampled = dataset.select(sampled_indices)
        tasks = [instance_to_task(row, hf_id) for row in sampled]
        for task in tasks:
            task["mix_source"] = "swe_smith_filtered_mix"
            task["mix_language"] = language
        all_tasks.extend(tasks)
        summary.append(
            {
                "source": hf_id,
                "language": language,
                "candidate_count": len(valid_indices),
                "selected_count": len(tasks),
            }
        )

    return all_tasks, summary


def load_rebench_tasks(dockerhub_username: str) -> list[dict[str, Any]]:
    print("\n--- Loading nebius/SWE-rebench-V2 ---")
    tasks = prepare_dataset(
        "swe_rebench_v2",
        split="train",
        dockerhub_username=dockerhub_username,
    )
    for task in tasks:
        lines = patch_line_count(task.get("patch"))
        log_length = log10_patch_length(lines)
        bucket = assign_bucket(log_length)
        language = task.get("language", "")
        task["patch_length_lines"] = lines
        task["log10_patch_length_lines"] = log_length
        task["length_bucket"] = bucket.name
        task["length_bucket_label"] = bucket.label
        task["mix_source"] = "swe_rebench_v2"
        task["mix_language"] = language
        task["data_source"] = REBENCH_DATASET_BY_LANGUAGE.get(language, f"swe_rebench_v2_{language}")
    return tasks


def group_rebench_tasks(tasks: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        language = task.get("language", "")
        bucket = task.get("length_bucket", "")
        grouped[language][bucket].append(task)
    return grouped


def add_bucket_selection(
    selected: list[dict[str, Any]],
    manifest_groups: list[dict[str, Any]],
    grouped: dict[str, dict[str, list[dict[str, Any]]]],
    language: str,
    bucket: LengthBucket,
    count: int,
    seed: int,
) -> None:
    candidates = grouped.get(language, {}).get(bucket.name, [])
    sample_seed = stable_sample_seed(seed, language, bucket.name, str(count))
    sampled = sample_tasks(candidates, count, sample_seed)
    print(f"  {language:>6} {bucket.label:>11}: selected {len(sampled)} of {len(candidates)} candidates")
    selected.extend(sampled)
    manifest_groups.append(
        {
            "source": "nebius/SWE-rebench-V2",
            "language": language,
            "bucket": bucket.name,
            "bucket_label": bucket.label,
            "sample_seed": sample_seed,
            "target_count": count,
            "candidate_count": len(candidates),
            "selected_count": len(sampled),
            "instance_ids": [task["instance_id"] for task in sampled],
        }
    )


def add_all_language(
    selected: list[dict[str, Any]],
    manifest_groups: list[dict[str, Any]],
    rebench_tasks: list[dict[str, Any]],
    language: str,
    repeat_count: int = 1,
) -> None:
    unique_sampled = sorted(
        [task for task in rebench_tasks if task.get("language") == language],
        key=lambda item: item["instance_id"],
    )
    sampled = repeat_task_rows(unique_sampled, repeat_count)
    if repeat_count == 1:
        print(f"  {language:>6} {'all':>11}: selected all {len(sampled)} instances")
    else:
        print(f"  {language:>6} {'all':>11}: selected all {len(unique_sampled)} instances x {repeat_count} = {len(sampled)} rows")
    selected.extend(sampled)
    manifest_groups.append(
        {
            "source": "nebius/SWE-rebench-V2",
            "language": language,
            "bucket": "all",
            "bucket_label": "all",
            "sample_seed": None,
            "target_count": None,
            "candidate_count": len(unique_sampled),
            "unique_selected_count": len(unique_sampled),
            "repeat_count": repeat_count,
            "selected_count": len(sampled),
            "unique_instance_ids": [task["instance_id"] for task in unique_sampled],
            "instance_ids": [task["instance_id"] for task in sampled],
        }
    )


def add_java_selection(
    selected: list[dict[str, Any]],
    manifest_groups: list[dict[str, Any]],
    grouped: dict[str, dict[str, list[dict[str, Any]]]],
    seed: int,
    target_count: int = 300,
) -> None:
    low = sorted(grouped.get("java", {}).get(BUCKET_0_1_5.name, []), key=lambda item: item["instance_id"])
    selected.extend(low)

    remaining = max(target_count - len(low), 0)
    mid_candidates = grouped.get("java", {}).get(BUCKET_1_5_2_0.name, [])
    sample_seed = stable_sample_seed(seed, "java", BUCKET_1_5_2_0.name, str(remaining))
    mid = sample_tasks(mid_candidates, remaining, sample_seed) if remaining else []
    selected.extend(mid)

    print(f"    java {BUCKET_0_1_5.label:>11}: selected all {len(low)} candidates; {BUCKET_1_5_2_0.label}: selected {len(mid)} of {len(mid_candidates)} to reach {len(low) + len(mid)}")
    manifest_groups.append(
        {
            "source": "nebius/SWE-rebench-V2",
            "language": "java",
            "bucket": f"{BUCKET_0_1_5.name}+{BUCKET_1_5_2_0.name}",
            "bucket_label": f"all {BUCKET_0_1_5.label} plus {BUCKET_1_5_2_0.label} to target",
            "sample_seed": sample_seed,
            "target_count": target_count,
            "candidate_count": len(low) + len(mid_candidates),
            "selected_count": len(low) + len(mid),
            "low_bucket_selected_count": len(low),
            "mid_bucket_selected_count": len(mid),
            "instance_ids": [task["instance_id"] for task in [*low, *mid]],
        }
    )


def select_rebench_mix(
    rebench_tasks: list[dict[str, Any]],
    seed: int,
    c_cpp_repeat_count: int = DEFAULT_C_CPP_REPEAT_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = group_rebench_tasks(rebench_tasks)
    selected: list[dict[str, Any]] = []
    manifest_groups: list[dict[str, Any]] = []

    print("\n--- Selecting SWE-rebench V2 rows ---")
    add_bucket_selection(selected, manifest_groups, grouped, "python", BUCKET_0_1_5, 300, seed)
    add_bucket_selection(selected, manifest_groups, grouped, "go", BUCKET_0_1_5, 300, seed)
    add_bucket_selection(selected, manifest_groups, grouped, "php", BUCKET_0_1_5, 300, seed)
    add_bucket_selection(selected, manifest_groups, grouped, "php", BUCKET_1_5_2_0, 300, seed)
    add_java_selection(selected, manifest_groups, grouped, seed, target_count=300)
    add_bucket_selection(selected, manifest_groups, grouped, "js", BUCKET_0_1_5, 300, seed)
    add_bucket_selection(selected, manifest_groups, grouped, "ts", BUCKET_0_1_5, 600, seed)
    add_all_language(selected, manifest_groups, rebench_tasks, "c", repeat_count=c_cpp_repeat_count)
    add_all_language(selected, manifest_groups, rebench_tasks, "cpp", repeat_count=c_cpp_repeat_count)

    return selected, manifest_groups


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def serialize_mix_fields(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized = serialize_complex_fields(tasks)
    for task in serialized:
        if isinstance(task.get("install_config"), (dict, list)):
            task["install_config"] = json.dumps(task["install_config"])
    return serialized


def prepare_swe_smith_rebenchv2_mix(
    seed: int = 42,
    swe_smith_max_per_dataset: int = 300,
    dataset_name: str | None = None,
    dataset_name_prefix: str = "swe_smith_rebenchv2",
    dockerhub_username: str = "jefzda",
    c_cpp_repeat_count: int = DEFAULT_C_CPP_REPEAT_COUNT,
    manifest_path: str | None = None,
    shuffle: bool = True,
    dry_run: bool = False,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    smith_tasks, smith_summary = load_swe_smith_filtered_tasks(
        max_per_dataset=swe_smith_max_per_dataset,
        seed=seed,
    )
    rebench_tasks = load_rebench_tasks(dockerhub_username=dockerhub_username)
    selected_rebench, rebench_summary = select_rebench_mix(
        rebench_tasks,
        seed=seed,
        c_cpp_repeat_count=c_cpp_repeat_count,
    )

    all_tasks = [*smith_tasks, *selected_rebench]
    if shuffle:
        random.Random(stable_sample_seed(seed, "final-shuffle")).shuffle(all_tasks)

    final_name = dataset_name or f"{dataset_name_prefix}_{len(all_tasks)}"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": final_name,
        "dataset_size": len(all_tasks),
        "seed": seed,
        "shuffle": shuffle,
        "log_patch_length": "log10(max(patch_line_count, 1))",
        "swe_smith": {
            "recipe": "swe_smith_filtered_mix",
            "max_per_dataset": swe_smith_max_per_dataset,
            "selected_count": len(smith_tasks),
            "groups": smith_summary,
        },
        "swe_rebench_v2": {
            "selected_count": len(selected_rebench),
            "c_cpp_repeat_count": c_cpp_repeat_count,
            "groups": rebench_summary,
        },
    }

    print(f"\n=== Combined total: {len(all_tasks)} instances ===")
    print(f"Dataset name: {final_name}")

    if manifest_path:
        json_dump(Path(manifest_path), manifest)
        print(f"Wrote manifest to {manifest_path}")

    if dry_run:
        print("Dry run: skipped DatasetRegistry registration")
        return final_name, all_tasks, manifest

    serialized = serialize_mix_fields(all_tasks)
    DatasetRegistry.register_dataset(
        final_name,
        serialized,
        source="SWE-smith trajectory mix + nebius/SWE-rebench-V2",
        description="SWE-smith filtered mix plus SWE-rebench V2 rows sampled by log patch length.",
        category="code",
    )
    print(f"Registered '{final_name}' in DatasetRegistry")

    loaded = DatasetRegistry.load_dataset(final_name)
    if loaded is None:
        raise RuntimeError(f"Dataset '{final_name}' was not found after registration")
    print(f"Verified: {len(loaded)} instances in registry")

    return final_name, all_tasks, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SWE-smith + SWE-rebench V2 training mix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument(
        "--swe-smith-max-per-dataset",
        type=int,
        default=300,
        help="Max SWE-smith instances per trajectory dataset (default: 300)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Exact DatasetRegistry name. Defaults to swe_smith_rebenchv2_{final_size}",
    )
    parser.add_argument(
        "--name-prefix",
        default="swe_smith_rebenchv2",
        help="DatasetRegistry name prefix when --name is omitted (default: swe_smith_rebenchv2)",
    )
    parser.add_argument(
        "--dockerhub-username",
        default="jefzda",
        help="DockerHub username used for datasets that need it (default: jefzda)",
    )
    parser.add_argument(
        "--c-cpp-repeat-count",
        type=int,
        default=DEFAULT_C_CPP_REPEAT_COUNT,
        help=f"Total copies for each selected C/CPP row (default: {DEFAULT_C_CPP_REPEAT_COUNT})",
    )
    parser.add_argument("--manifest", default=None, help="Optional path to write a JSON manifest")
    parser.add_argument("--no-shuffle", action="store_true", help="Preserve source grouping order")
    parser.add_argument("--dry-run", action="store_true", help="Build and summarize, but do not register")
    args = parser.parse_args()

    prepare_swe_smith_rebenchv2_mix(
        seed=args.seed,
        swe_smith_max_per_dataset=args.swe_smith_max_per_dataset,
        dataset_name=args.name,
        dataset_name_prefix=args.name_prefix,
        dockerhub_username=args.dockerhub_username,
        c_cpp_repeat_count=args.c_cpp_repeat_count,
        manifest_path=args.manifest,
        shuffle=not args.no_shuffle,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

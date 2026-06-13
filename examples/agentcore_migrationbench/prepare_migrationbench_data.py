"""Prepare MigrationBench dataset for rLLM training with AgentCore runtime.

Prerequisite: run the strands_migration_agent preprocess step first to upload the
MigrationBench repos to S3. We upload the repos rather than loading them from
GitHub at training time so the dataset remains available even if an upstream
repo is deleted or made private.

    https://github.com/awslabs/agentcore-rl-toolkit/tree/main/examples/strands_migration_agent

    cd <agentcore-rl-toolkit>/examples/strands_migration_agent
    python preprocess.py --s3-bucket-name <your-bucket>

This script then downloads only the tiny metadata.json files (~241 bytes each)
from that same bucket to filter by num_test_cases, and registers train/test
splits with the rLLM DatasetRegistry. The actual repo tarballs stay in S3 and
are pulled at runtime by the AgentCore container.

Train: repos from s3://<bucket>/tars/train/ with num_test_cases > 0
Test:  all repos from s3://<bucket>/tars/test/

Usage:
    python -m examples.agentcore_migrationbench.prepare_migrationbench_data \\
        --s3-bucket-name <your-bucket>
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from rllm.data.dataset import DatasetRegistry

DATASET_NAME = "migration_bench"
PROMPT_TEMPLATE = "Please help migrate this repo: {repo_path}. There are {num_tests} test cases in it."


def list_s3_folders(s3_bucket: str, s3_prefix: str) -> list[str]:
    """List immediate sub-folder names under an S3 prefix."""
    result = subprocess.run(
        ["aws", "s3", "ls", f"s3://{s3_bucket}/{s3_prefix}"],
        capture_output=True,
        text=True,
        check=True,
    )
    folders = []
    for line in result.stdout.strip().splitlines():
        parts = line.strip().split()
        if parts[0] == "PRE":
            folders.append(parts[1].rstrip("/"))
    return folders


def download_all_metadata(s3_bucket: str, s3_prefix: str, folders: list[str], dest_dir: Path) -> None:
    """Download metadata.json for all folders in parallel."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    commands = "\n".join(f"aws s3 cp s3://{s3_bucket}/{s3_prefix}{f}/metadata.json {dest_dir}/{f}.json --quiet" for f in folders)
    subprocess.run(
        ["xargs", "-P", "32", "-I", "{}", "bash", "-c", "{}"],
        input=commands,
        text=True,
        check=False,
    )


def build_records(s3_bucket: str, metadata_dir: Path, s3_prefix: str, filter_positive_tests: bool) -> list[dict]:
    """Load metadata and build dataset records."""
    records = []
    for f in sorted(metadata_dir.glob("*.json")):
        try:
            with open(f) as fh:
                meta = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        if filter_positive_tests and meta.get("num_test_cases", 0) <= 0:
            continue

        folder_name = f.stem
        records.append(
            {
                "prompt": PROMPT_TEMPLATE,
                "repo_uri": f"s3://{s3_bucket}/{s3_prefix}{folder_name}/{folder_name}.tar.gz",
                "metadata_uri": f"s3://{s3_bucket}/{s3_prefix}{folder_name}/metadata.json",
                "require_maximal_migration": False,
                "data_source": "migration_bench",
                "num_test_cases": meta.get("num_test_cases", 0),
                "num_loc": meta.get("num_loc", 0),
                "repo": meta.get("repo", folder_name),
            }
        )
    return records


def print_stats(records: list[dict], split: str):
    """Print summary statistics."""
    test_counts = [r["num_test_cases"] for r in records]
    pos = [t for t in test_counts if t > 0]

    print(f"\n  {split.upper()}: {len(records)} repos")
    if not pos:
        return

    buckets = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 500), (501, float("inf"))]
    for lo, hi in buckets:
        label = f"{lo}-{int(hi)}" if hi != float("inf") else f"{lo}+"
        count = sum(1 for t in pos if lo <= t <= hi)
        if count:
            print(f"    tests {label:<10} {count:>5} ({count / len(pos) * 100:.1f}%)")

    neg = sum(1 for t in test_counts if t <= 0)
    if neg:
        print(f"    tests <= 0 (filtered): {neg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-bucket-name", required=True)
    args = parser.parse_args()
    s3_bucket = args.s3_bucket_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Train
        print("Fetching train metadata...")
        train_folders = list_s3_folders(s3_bucket, "tars/train/")
        print(f"  {len(train_folders)} repos found")
        download_all_metadata(s3_bucket, "tars/train/", train_folders, tmpdir / "train")
        train_records = build_records(s3_bucket, tmpdir / "train", "tars/train/", filter_positive_tests=True)
        print_stats(train_records, "train")

        # Test
        print("\nFetching test metadata...")
        test_folders = list_s3_folders(s3_bucket, "tars/test/")
        print(f"  {len(test_folders)} repos found")
        download_all_metadata(s3_bucket, "tars/test/", test_folders, tmpdir / "test")
        test_records = build_records(s3_bucket, tmpdir / "test", "tars/test/", filter_positive_tests=False)
        print_stats(test_records, "test")

    # Register
    print("\nRegistering datasets...")
    DatasetRegistry.register_dataset(
        name=DATASET_NAME,
        data=train_records,
        split="train",
        source="AmazonScience/migration-bench-java-full",
        description=f"Java 8→17 migration benchmark, train ({len(train_records)} repos with tests > 0)",
        category="code",
    )
    DatasetRegistry.register_dataset(
        name=DATASET_NAME,
        data=test_records,
        split="test",
        source="AmazonScience/migration-bench-java-selected",
        description=f"Java 8→17 migration benchmark, test ({len(test_records)} repos)",
        category="code",
    )
    print(f"  {DATASET_NAME}/train: {len(train_records)} examples")
    print(f"  {DATASET_NAME}/test:  {len(test_records)} examples")
    print(f'\nLoad with: DatasetRegistry.load_dataset("{DATASET_NAME}", "train")')


if __name__ == "__main__":
    main()

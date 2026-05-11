"""Prepare and register the deepcoder train/test datasets.

Pulls the three training subsets (primeintellect, taco, lcbv5) and the
two test subsets (codeforces, lcbv5) of ``agentica-org/DeepCoder-Preview-Dataset``
and normalizes them into rows the cookbook's flow + evaluator can consume.

Usage::

    python cookbooks/deepcoder/prepare_deepcoder_data.py
    # or, faster smoke run:
    python cookbooks/deepcoder/prepare_deepcoder_data.py --train-size 200 --test-size 50
"""

from __future__ import annotations

import argparse
import json

from datasets import concatenate_datasets, load_dataset

from rllm.data.dataset import DatasetRegistry
from rllm.data.utils import fetch_live_code_bench_system_prompt


def _normalize_tests(tests_raw):
    """Normalize the various Deepcoder test schemas into a flat list of dicts.

    - Codeforces / LiveCodeBench: already a list of {input, output, testtype}.
    - TACO / APPS: a dict {"inputs": [...], "outputs": [...]}.
    """
    if isinstance(tests_raw, str):
        tests = json.loads(tests_raw)
    else:
        tests = tests_raw
    if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
        return [{"input": i, "output": o, "testtype": "stdin_stdout"} for i, o in zip(tests["inputs"], tests["outputs"], strict=False)]
    if isinstance(tests, list):
        return tests
    return [tests] if tests else []


def _preprocess(example: dict, idx: int) -> dict:
    starter_code = example.get("starter_code", "") or ""
    question = fetch_live_code_bench_system_prompt(example["problem"], starter_code or None)

    metadata = example.get("metadata", {}) or {}
    tests = _normalize_tests(example.get("tests"))

    for t in tests:
        if t.get("testtype") == "functional" and metadata.get("func_name") is not None:
            t["metadata"] = {"func_name": str(metadata["func_name"])}
        else:
            t["metadata"] = {"func_name": None}

    return {
        "uid": f"deepcoder_{idx}",
        "index": idx,
        "question": question,
        "ground_truth": json.dumps(tests),
        "data_source": "livecodebench",
        "starter_code": starter_code,
        "metadata": json.dumps(metadata),
    }


def prepare_deepcoder_data(train_size: int | None = None, test_size: int | None = None):
    train_ds = concatenate_datasets(
        [
            load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="primeintellect", split="train"),
            load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="taco", split="train"),
            load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="train"),
        ]
    )
    test_ds = concatenate_datasets(
        [
            load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="codeforces", split="test"),
            load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="test"),
        ]
    )

    if train_size:
        train_ds = train_ds.select(range(min(train_size, len(train_ds))))
    if test_size:
        test_ds = test_ds.select(range(min(test_size, len(test_ds))))

    train_ds = train_ds.map(_preprocess, with_indices=True, writer_batch_size=10, num_proc=16)
    test_ds = test_ds.map(_preprocess, with_indices=True, writer_batch_size=10, num_proc=16)

    train = DatasetRegistry.register_dataset("deepcoder", train_ds, "train")
    test = DatasetRegistry.register_dataset("deepcoder", test_ds, "test")
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=None, help="Cap train rows (default: full).")
    ap.add_argument("--test-size", type=int, default=None, help="Cap test rows (default: full).")
    args = ap.parse_args()

    train, test = prepare_deepcoder_data(train_size=args.train_size, test_size=args.test_size)
    print(f"Train: {len(train.get_data())} rows")
    print(f"Test:  {len(test.get_data())} rows")
    print("Sample train row:", train.get_data()[0])


if __name__ == "__main__":
    main()

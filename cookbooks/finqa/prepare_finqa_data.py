"""Download and register the rLLM/finqa dataset.

Pulls the tarball from HuggingFace, extracts the company-tables tree
(~6,900 tables across 207 companies) into ``cookbooks/finqa/data/``,
and registers the train / val / test question splits with rLLM's
``DatasetRegistry``.

Usage::

    python cookbooks/finqa/prepare_finqa_data.py
"""

from __future__ import annotations

import json
import tarfile

import finqa_constants as C
import pandas as pd
from huggingface_hub import hf_hub_download

from rllm.data.dataset import DatasetRegistry

HF_REPO_ID = "rLLM/finqa"
HF_FILENAME = "data.tar.gz"


def download_data() -> None:
    """Fetch the finqa tarball from HF if the data dir is missing."""
    data_dir = C.DATA_DIR
    if data_dir.exists() and (data_dir / "train_finqa.csv").exists():
        print(f"Data already exists at {data_dir}")
        return

    print(f"Downloading finqa data from {HF_REPO_ID}...")
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, repo_type="dataset")
    print(f"Extracting to {data_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir.parent)
    print("Done.")


def _parse_json_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = s
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, str):
            cleaned = parsed.strip()
            return [cleaned] if cleaned else []
        return []
    return []


def prepare_finqa_data():
    train_df = pd.read_csv(C.TRAIN_QUESTIONS_PATH)
    val_df = pd.read_csv(C.VAL_QUESTIONS_PATH)
    test_df = pd.read_csv(C.TEST_QUESTIONS_PATH)

    def preprocess_fn(example: dict) -> dict:
        return {
            "question": example["user_query"],
            "ground_truth": example["answer"],
            "data_source": "finqa",
            "company": example["company"],
            "question_id": str(example["id"]),
            "question_type": example["question_type"],
            "core_question": example["question"],
            "table_name": _parse_json_list(example.get("table_name")),
            "columns_used": _parse_json_list(example.get("columns_used_json")),
            "rows_used": _parse_json_list(example.get("rows_used_json")),
            "explanation": example["explanation"],
        }

    train = [preprocess_fn(row) for _, row in train_df.iterrows()]
    val = [preprocess_fn(row) for _, row in val_df.iterrows()]
    test = [preprocess_fn(row) for _, row in test_df.iterrows()]

    train_dataset = DatasetRegistry.register_dataset("finqa", train, "train")
    val_dataset = DatasetRegistry.register_dataset("finqa", val, "val")
    test_dataset = DatasetRegistry.register_dataset("finqa", test, "test")
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    download_data()
    train, val, test = prepare_finqa_data()
    print(f"Train: {len(train.get_data())} rows")
    print(f"Val:   {len(val.get_data())} rows")
    print(f"Test:  {len(test.get_data())} rows")

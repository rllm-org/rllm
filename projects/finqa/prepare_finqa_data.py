import json
import tarfile

import pandas as pd
from huggingface_hub import hf_hub_download

from projects.finqa import constants as C
from rllm.data.dataset import DatasetRegistry

HF_REPO_ID = "rLLM/finqa"
HF_FILENAME = "data.tar.gz"


def download_data():
    """Download and extract finqa data from HuggingFace if not present."""
    data_dir = C.DATA_DIR

    # Check if data already exists
    if data_dir.exists() and (data_dir / "train_finqa.csv").exists():
        print(f"Data already exists at {data_dir}")
        return

    print(f"Downloading finqa data from {HF_REPO_ID}...")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download tar.gz from HuggingFace
    tar_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=HF_FILENAME, repo_type="dataset"
    )

    # Extract to parent directory (tar contains data/ prefix)
    print(f"Extracting to {data_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir.parent)

    print("Done.")


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _parse_json_list(value):
    """Decode columns stored as JSON strings, defaulting to [] for empty values."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = stripped
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, str):
            cleaned = parsed.strip()
            return [cleaned] if cleaned else []
        return []
    return []


def prepare_finqa_data():
    train_df = _load_csv(C.TRAIN_QUESTIONS_PATH)
    val_df = _load_csv(C.VAL_QUESTIONS_PATH)
    test_df = _load_csv(C.TEST_QUESTIONS_PATH)

    def preprocess_fn(example):
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

    train_processed = [preprocess_fn(row) for _, row in train_df.iterrows()]
    val_processed = [preprocess_fn(row) for _, row in val_df.iterrows()]
    test_processed = [preprocess_fn(row) for _, row in test_df.iterrows()]

    train_dataset = DatasetRegistry.register_dataset("finqa", train_processed, "train")
    val_dataset = DatasetRegistry.register_dataset("finqa", val_processed, "val")
    test_dataset = DatasetRegistry.register_dataset("finqa", test_processed, "test")
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    download_data()
    train_dataset, val_dataset, test_dataset = prepare_finqa_data()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Train dataset path: {train_dataset.get_data_path()}")
    print(f"Validation dataset path: {val_dataset.get_data_path()}")
    print(f"Test dataset path: {test_dataset.get_data_path()}")

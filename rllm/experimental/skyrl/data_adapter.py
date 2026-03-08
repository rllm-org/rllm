"""
Data adapter utilities for the SkyRL backend.

This module normalizes rLLM datasets into SkyRL-compatible records and, when
needed, materializes temporary parquet files for SkyRL PromptDataset loading.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from rllm.data import Dataset

# Priority order for prompt keys
PROMPT_KEYS = ["prompt", "question", "problem"]

# Reserved keys that should not be included in env_extras
RESERVED_KEYS = {"prompt", "question", "problem", "env_class", "uid", "unique_id", "data_source"}


@dataclass(frozen=True)
class SkyRLDatasetFile:
    """A dataset file reference used to build SkyRL PromptDataset."""

    path: str
    cleanup_required: bool = False


def _is_chat_prompt(prompt: Any) -> bool:
    return isinstance(prompt, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt)


def _is_skyrl_compatible_dataset(data: list[dict[str, Any]]) -> bool:
    """Return True when records are already in SkyRL PromptDataset-ready shape."""
    if not data:
        return True
    for item in data:
        if not _is_chat_prompt(item.get("prompt")):
            return False
        if item.get("env_class") is None:
            return False
    return True


def adapt_single_item(item: dict[str, Any]) -> dict[str, Any]:
    """Adapt a single rLLM dataset item to SkyRL format."""
    prompt = None
    prompt_key = None

    for key in PROMPT_KEYS:
        if key in item and item[key] is not None:
            prompt = item[key]
            prompt_key = key
            break

    if prompt is None:
        raise ValueError(f"Cannot find prompt in item. Expected one of {PROMPT_KEYS}, but found keys: {list(item.keys())}")

    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        if not _is_chat_prompt(prompt):
            raise ValueError(f"Prompt in list format must be a list of dicts with 'role' and 'content' keys. Got: {prompt}")
    else:
        raise ValueError(f"Prompt must be either a string or a list of dicts. Got type: {type(prompt)}, value: {prompt}")

    env_class = item.get("env_class")
    if env_class is None:
        env_class = item.get("data_source")

    uid = item.get("uid")
    if uid is None:
        uid = item.get("unique_id")

    env_extras = {}
    for key, value in item.items():
        if key not in RESERVED_KEYS and key != prompt_key:
            env_extras[key] = value

    if prompt_key:
        env_extras["_rllm_original_prompt_key"] = prompt_key
        if isinstance(item[prompt_key], str):
            env_extras["_rllm_original_prompt_value"] = item[prompt_key]

    adapted_item = {
        "prompt": prompt,
        "env_extras": env_extras,
    }

    if env_class is not None:
        adapted_item["env_class"] = env_class
    if uid is not None:
        adapted_item["uid"] = uid

    return adapted_item


def adapt_rllm_batch_to_skyrl(batch: list[dict[str, Any]], default_env_class: str | None = None) -> list[dict[str, Any]]:
    """Adapt a batch of rLLM dataset items to SkyRL format."""
    if not batch:
        return []

    adapted = [adapt_single_item(item) for item in batch]
    if default_env_class is not None:
        for item in adapted:
            item.setdefault("env_class", default_env_class)
    return adapted


def prepare_skyrl_dataset_file(dataset: Dataset | None, default_env_class: str = "BaseTextEnv") -> SkyRLDatasetFile | None:
    """Return a PromptDataset-ready parquet path for SkyRL.

    Reuses the existing registered parquet file when records are already
    SkyRL-compatible. Otherwise creates a temporary converted parquet file.
    """
    if dataset is None:
        return None

    data = dataset.get_data()
    dataset_path = dataset.get_data_path()
    if dataset_path and os.path.exists(dataset_path) and _is_skyrl_compatible_dataset(data):
        return SkyRLDatasetFile(path=dataset_path, cleanup_required=False)

    converted_data = adapt_rllm_batch_to_skyrl(data, default_env_class=default_env_class)

    temp_fd, temp_path = tempfile.mkstemp(suffix=".parquet")
    os.close(temp_fd)
    try:
        pd.DataFrame(converted_data).to_parquet(temp_path)
    except Exception as exc:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise RuntimeError(f"Failed to convert rLLM Dataset to SkyRL format: {exc}") from exc

    return SkyRLDatasetFile(path=temp_path, cleanup_required=True)


def cleanup_temporary_dataset_files(dataset_files: Iterable[SkyRLDatasetFile | None]) -> None:
    """Remove temporary parquet files created by prepare_skyrl_dataset_file."""
    for dataset_file in dataset_files:
        if dataset_file is None:
            continue
        if dataset_file.cleanup_required and os.path.exists(dataset_file.path):
            os.unlink(dataset_file.path)

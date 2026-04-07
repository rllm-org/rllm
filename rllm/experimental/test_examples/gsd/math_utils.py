"""Math dataset and reward utilities for GSD experiments.

Loads the DeepScaleR training set and AIME 2024 test set from the
rLLM dataset registry.  Both must be pre-registered by running::

    python -m examples.deepscaler.prepare_math_data
"""

from __future__ import annotations

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn


def prepare_deepscaler_datasets():
    """Load pre-registered DeepScaleR (train) + AIME 2024 (test) datasets.

    Raises:
        RuntimeError: If datasets are not registered.
    """
    train = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test = DatasetRegistry.load_dataset("aime2024", "test")
    if train is None or test is None:
        raise RuntimeError("Datasets not found. Run:\n  python -m examples.deepscaler.prepare_math_data\nto register deepscaler_math and aime2024.")
    return train, test


def gsd_math_reward(task: dict, response_text: str) -> float:
    """Binary math reward: 1.0 if correct, 0.0 otherwise."""
    result = math_reward_fn(task, response_text)
    return 1.0 if result.is_correct else 0.0

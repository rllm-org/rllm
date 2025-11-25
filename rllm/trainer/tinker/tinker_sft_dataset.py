"""Tinker-specific SFT dataset for rLLM.

This module provides a dataset class that converts rLLM's parquet-based
training data into Tinker's Datum format for supervised fine-tuning.
"""

from __future__ import annotations

import logging

import datasets
import tinker
from tinker_cookbook.renderers import Message, Renderer, TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.types import SupervisedDataset

logger = logging.getLogger(__name__)


def conversation_to_datum(
    conversation: list[Message],
    renderer: Renderer,
    max_length: int | None,
    train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
) -> tinker.Datum:
    """Convert a conversation (list of messages) to a Tinker Datum."""
    tokens, weights = renderer.build_supervised_example(conversation, train_on_what=train_on_what)
    return datum_from_tokens_weights(tokens, weights, max_length)


class TinkerSFTDataset(SupervisedDataset):
    """
    Dataset for Tinker SFT that loads from rLLM sources.

    This dataset:
    - Loads from parquet files OR HuggingFace Dataset objects (from DatasetRegistry)
    - Uses Tinker's Renderer system for tokenization
    - Produces Tinker Datum objects for training
    - Supports batching and epoch shuffling
    """

    def __init__(
        self,
        dataset_or_files: datasets.Dataset | str | list[str],
        renderer: Renderer,
        batch_size: int,
        max_length: int | None = None,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        max_samples: int = -1,
    ):
        """
        Initialize the Tinker SFT dataset.

        Args:
            dataset_or_files: Either a HuggingFace Dataset object (from DatasetRegistry)
                            or path(s) to parquet file(s) containing "messages" field
            renderer: Tinker renderer for tokenization
            batch_size: Number of examples per batch
            max_length: Maximum sequence length (None for no limit)
            train_on_what: Which tokens to apply loss to
            max_samples: Maximum number of samples to use (-1 for all)
        """
        self.renderer = renderer
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_on_what = train_on_what

        # Load dataset from either Dataset object or parquet files
        # Check if it's a string or list (file paths), otherwise assume it's a Dataset object
        if isinstance(dataset_or_files, str | list):
            # Parquet file paths provided
            if isinstance(dataset_or_files, str):
                dataset_or_files = [dataset_or_files]
            self.dataset = datasets.load_dataset("parquet", data_files=dataset_or_files, split="train")
            source = dataset_or_files
        else:
            # Dataset object provided directly (e.g., from DatasetRegistry)
            # This handles both HuggingFace datasets and rLLM datasets (which now have .shuffle() and .select())
            self.dataset = dataset_or_files
            source = "Dataset object"

        # Limit samples if specified (all datasets now have .select() method)
        if max_samples > 0 and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))
            logger.info(f"Limited dataset to {max_samples} samples")

        logger.info(f"Loaded {len(self.dataset)} examples from {source}")
        logger.info(f"Training on: {train_on_what}")

    def get_batch(self, index: int) -> list[tinker.Datum]:
        """Get a batch of Tinker Datums."""
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataset))

        datums = []
        for i in range(start_idx, end_idx):
            row = self.dataset[i]
            messages = row["messages"]
            datum = conversation_to_datum(
                messages,
                self.renderer,
                self.max_length,
                self.train_on_what,
            )
            datums.append(datum)

        return datums

    def set_epoch(self, seed: int = 0):
        """Shuffle the dataset for a new epoch (all datasets now have .shuffle() method)."""
        self.dataset = self.dataset.shuffle(seed=seed)
        logger.info(f"Shuffled dataset with seed {seed} ({len(self.dataset)} samples)")

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.dataset) // self.batch_size


def create_tinker_sft_datasets(
    train_data: datasets.Dataset | str | list[str],
    val_data: datasets.Dataset | str | list[str] | None,
    renderer: Renderer,
    batch_size: int,
    val_batch_size: int | None = None,
    max_length: int | None = None,
    train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    max_train_samples: int = -1,
    max_val_samples: int = -1,
) -> tuple[TinkerSFTDataset, TinkerSFTDataset | None]:
    """
    Create train and optional validation datasets for Tinker SFT.

    Args:
        train_data: Either a HuggingFace Dataset (from DatasetRegistry) or path(s) to training parquet file(s)
        val_data: Either a HuggingFace Dataset (from DatasetRegistry) or path(s) to validation parquet file(s) (optional)
        renderer: Tinker renderer for tokenization
        batch_size: Training batch size
        val_batch_size: Validation batch size (defaults to batch_size)
        max_length: Maximum sequence length
        train_on_what: Which tokens to apply loss to
        max_train_samples: Maximum training samples (-1 for all)
        max_val_samples: Maximum validation samples (-1 for all)

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    if val_batch_size is None:
        val_batch_size = batch_size

    train_dataset = TinkerSFTDataset(
        dataset_or_files=train_data,
        renderer=renderer,
        batch_size=batch_size,
        max_length=max_length,
        train_on_what=train_on_what,
        max_samples=max_train_samples,
    )

    val_dataset = None
    if val_data:
        val_dataset = TinkerSFTDataset(
            dataset_or_files=val_data,
            renderer=renderer,
            batch_size=val_batch_size,
            max_length=max_length,
            train_on_what=train_on_what,
            max_samples=max_val_samples,
        )

    return train_dataset, val_dataset

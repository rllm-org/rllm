"""Tinker SFT dataset: converts rLLM ``messages`` rows into Tinker Datums.

Migrated from the (removed) ``rllm.trainer.deprecated.tinker_sft_dataset``.
Imported lazily by :class:`rllm.trainer.sft.tinker_backend.TinkerSFTBackend`, so
``tinker``/``tinker_cookbook`` are only required when actually training on tinker.
"""

from __future__ import annotations

import logging

import datasets
import tinker
from tinker_cookbook.renderers import Message, Renderer, TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDataset

logger = logging.getLogger(__name__)


def _ensure_trainable(conversation: list[Message], last_only: bool) -> list[Message]:
    """Ensure every message carries a ``trainable`` flag, so the renderer can
    always use tinker's ``CUSTOMIZED`` mode — the data alone decides the mask.

    Self-describing rows (e.g. from ``from-eval``'s automerge) already carry the
    flag and are returned untouched. A row without flags gets a derived default:
    assistant messages train (only the *last* when ``last_only``), reproducing the
    legacy ``ALL_ASSISTANT_MESSAGES`` / ``LAST_ASSISTANT_MESSAGE`` behavior.
    """
    if conversation and isinstance(conversation[0], dict) and "trainable" in conversation[0]:
        return conversation
    last_asst = max((i for i, m in enumerate(conversation) if m.get("role") == "assistant"), default=-1)
    return [{**m, "trainable": m.get("role") == "assistant" and (not last_only or i == last_asst)} for i, m in enumerate(conversation)]


def conversation_to_datum(
    conversation: list[Message],
    renderer: Renderer,
    max_length: int | None,
    last_only: bool = False,
) -> tinker.Datum:
    """Convert a conversation (list of messages) to a Tinker Datum.

    Always renders with ``CUSTOMIZED`` masking, driven by each message's
    ``trainable`` flag (derived via :func:`_ensure_trainable` when absent).
    """
    conversation = _ensure_trainable(conversation, last_only)
    model_input, weights = renderer.build_supervised_example(conversation, train_on_what=TrainOnWhat.CUSTOMIZED)
    return datum_from_model_input_weights(model_input, weights, max_length)


class TinkerSFTDataset(SupervisedDataset):
    """Dataset for Tinker SFT that loads from rLLM sources.

    Accepts a HuggingFace/rLLM Dataset object (from DatasetRegistry) or parquet
    file path(s) with a ``messages`` column, renders via Tinker's renderer, and
    yields Tinker Datums in batches.
    """

    def __init__(
        self,
        dataset_or_files: datasets.Dataset | str | list[str],
        renderer: Renderer,
        batch_size: int,
        max_length: int | None = None,
        last_only: bool = False,
        max_samples: int = -1,
    ):
        self.renderer = renderer
        self.batch_size = batch_size
        self.max_length = max_length
        self.last_only = last_only

        if isinstance(dataset_or_files, str | list):
            if isinstance(dataset_or_files, str):
                dataset_or_files = [dataset_or_files]
            self.dataset = datasets.load_dataset("parquet", data_files=dataset_or_files, split="train")
            source = dataset_or_files
        else:
            # Dataset object provided directly (HF or rLLM Dataset, both have
            # .shuffle()/.select()/__getitem__).
            self.dataset = dataset_or_files
            source = "Dataset object"

        if max_samples > 0 and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))
            logger.info(f"Limited dataset to {max_samples} samples")

        logger.info(f"Loaded {len(self.dataset)} examples from {source}")
        logger.info(f"Masking: CUSTOMIZED (derive last_only={last_only} for flag-less rows)")

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        datums = []
        for i in range(start_idx, end_idx):
            row = self.dataset[i]
            datums.append(conversation_to_datum(row["messages"], self.renderer, self.max_length, self.last_only))
        return datums

    def set_epoch(self, seed: int = 0):
        self.dataset = self.dataset.shuffle(seed=seed)
        logger.info(f"Shuffled dataset with seed {seed} ({len(self.dataset)} samples)")

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size


def create_tinker_sft_datasets(
    train_data: datasets.Dataset | str | list[str],
    val_data: datasets.Dataset | str | list[str] | None,
    renderer: Renderer,
    batch_size: int,
    val_batch_size: int | None = None,
    max_length: int | None = None,
    last_only: bool = False,
    max_train_samples: int = -1,
    max_val_samples: int = -1,
) -> tuple[TinkerSFTDataset, TinkerSFTDataset | None]:
    """Create train and optional validation datasets for Tinker SFT."""
    if val_batch_size is None:
        val_batch_size = batch_size

    train_dataset = TinkerSFTDataset(
        dataset_or_files=train_data,
        renderer=renderer,
        batch_size=batch_size,
        max_length=max_length,
        last_only=last_only,
        max_samples=max_train_samples,
    )

    val_dataset = None
    if val_data:
        val_dataset = TinkerSFTDataset(
            dataset_or_files=val_data,
            renderer=renderer,
            batch_size=val_batch_size,
            max_length=max_length,
            last_only=last_only,
            max_samples=max_val_samples,
        )

    return train_dataset, val_dataset

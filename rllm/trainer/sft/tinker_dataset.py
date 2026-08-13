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

from rllm.data.sft_schema import SFTSchemaError, normalize_messages
from rllm.trainer.sft.backend import SFTConfigError

logger = logging.getLogger(__name__)


def _ensure_trainable(conversation: list[Message], last_only: bool) -> list[Message]:
    """Legacy trainable-flag derivation (superseded by :mod:`rllm.data.sft_schema`).

    Retained as an importable helper for callers/tests that still reference it;
    the render path below now normalizes via the schema instead. Self-describing
    rows (e.g. from ``from-eval``'s automerge) already carry the flag and are
    returned untouched. A row without flags gets a derived default: assistant
    messages train (only the *last* when ``last_only``), reproducing the legacy
    ``ALL_ASSISTANT_MESSAGES`` / ``LAST_ASSISTANT_MESSAGE`` behavior.
    """
    if conversation and isinstance(conversation[0], dict) and "trainable" in conversation[0]:
        return conversation
    last_asst = max((i for i, m in enumerate(conversation) if m.get("role") == "assistant"), default=-1)
    return [{**m, "trainable": m.get("role") == "assistant" and (not last_only or i == last_asst)} for i, m in enumerate(conversation)]


def _row_context(conversation, limit: int = 400) -> str:
    """A truncated repr of a failing conversation, for actionable errors."""
    text = repr(conversation)
    return text if len(text) <= limit else text[:limit] + "..."


# Process-wide truncation-warning counter: warn loudly on the first few
# over-length rows, then thin out so a fully over-length dataset doesn't flood
# the logs (one reminder every 1000 rows).
_truncation_warn_count = 0


def _warn_truncation(row_tokens: int, max_length: int) -> None:
    global _truncation_warn_count
    _truncation_warn_count += 1
    if _truncation_warn_count <= 5 or _truncation_warn_count % 1000 == 0:
        logger.warning(
            "SFT row renders to %d tokens > data.max_length=%d: the datum is TRUNCATED and its tail — including "
            "the final trainable turn(s) — is dropped from training. If that is not intended, raise --max-length "
            "to at least %d (SFTSpec.max_length defaults to 2048, far below typical multi-turn trajectories). "
            "[over-length row #%d%s]",
            row_tokens,
            max_length,
            row_tokens,
            _truncation_warn_count,
            "; further warnings thinned to every 1000th" if _truncation_warn_count == 5 else "",
        )


def conversation_to_datum(
    conversation: list[Message],
    renderer: Renderer,
    max_length: int | None,
    last_only: bool = False,
    *,
    tools: list[dict] | None = None,
) -> tinker.Datum:
    """Convert a conversation (list of messages) to a Tinker Datum.

    Normalizes the raw messages through :mod:`rllm.data.sft_schema` (str/None
    content coercion, parquet ``None``-artifact stripping, structured
    ``tool_calls``, trainable-flag derivation) and renders with tinker's
    ``CUSTOMIZED`` masking — each message's ``trainable`` flag alone decides the
    loss mask. ``last_only`` selects the flag-less default (train just the last
    assistant turn) rather than the all-assistant default.

    Schema/validation failures are re-raised as :class:`SFTConfigError` with the
    failing row's context.
    """
    default_trainable = "last" if last_only else "all"
    try:
        messages = normalize_messages(conversation, default_trainable=default_trainable)
        tinker_messages = [m.to_tinker_message() for m in messages]
    except SFTSchemaError as e:
        raise SFTConfigError(f"SFT row failed schema normalization: {e}\n  row={_row_context(conversation)}") from e
    if not getattr(renderer, "strip_thinking_from_history", False):
        from rllm.renderers.adapters import prepare_tinker_messages_for_history

        tinker_messages = prepare_tinker_messages_for_history(
            renderer,
            tinker_messages,
            lift_reasoning=False,
        )

    if tools and hasattr(renderer, "build_supervised_example_with_tools"):
        model_input, weights = renderer.build_supervised_example_with_tools(
            tinker_messages,
            tools,
            train_on_what=TrainOnWhat.CUSTOMIZED,
        )
    else:
        if tools:
            from rllm.renderers.adapters import prepare_tinker_messages_with_tools

            try:
                tinker_messages = prepare_tinker_messages_with_tools(
                    renderer,
                    tinker_messages,
                    tools,
                )
            except (TypeError, ValueError) as e:
                raise SFTConfigError(f"SFT row has invalid tool declarations: {e}") from e
            for message in tinker_messages:
                message.setdefault("trainable", False)
        model_input, weights = renderer.build_supervised_example(
            tinker_messages,
            train_on_what=TrainOnWhat.CUSTOMIZED,
        )
    if max_length is not None and model_input.length > max_length:
        _warn_truncation(model_input.length, max_length)
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
            try:
                datums.append(
                    conversation_to_datum(
                        row["messages"],
                        self.renderer,
                        self.max_length,
                        self.last_only,
                        tools=row.get("tools"),
                    )
                )
            except SFTConfigError as e:
                raise SFTConfigError(f"dataset row {i}: {e}") from e
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

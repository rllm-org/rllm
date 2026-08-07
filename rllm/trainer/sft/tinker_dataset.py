"""Tinker SFT dataset: converts rLLM ``messages`` rows into Tinker Datums.

Migrated from the (removed) ``rllm.trainer.deprecated.tinker_sft_dataset``.
Imported lazily by :class:`rllm.trainer.sft.tinker_backend.TinkerSFTBackend`, so
``tinker``/``tinker_cookbook`` are only required when actually training on tinker.
"""

from __future__ import annotations

import json
import logging
import math
import random
from typing import Literal

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


def _message_byte_length(messages) -> int:
    """Cheap per-row length proxy: serialized byte length of the messages.

    Mirrors the Fireworks training cookbook's ``group_by_length`` proxy (raw
    JSONL byte length) — no tokenizer pass, and byte length tracks rendered
    token count closely for text-only rows (poor for base64 image parts, which
    this SFT path does not carry).
    """
    try:
        return len(json.dumps(messages, default=str))
    except (TypeError, ValueError):
        return len(str(messages))


def length_grouped_order(lengths: list[int], batch_size: int, factor: int, seed: int) -> list[int]:
    """Row ordering that packs similarly-sized rows into the same batch.

    Bucket-then-shuffle (the standard length-grouped sampler, matching the
    Fireworks cookbook's ``group_by_length``): a fresh seeded permutation is cut
    into windows of ``batch_size * factor`` rows, each window is sorted by length
    descending, then flattened. Contiguous ``batch_size`` slices of the result
    therefore hold near-equal-length rows, so a batch pads to little more than
    its own rows' real length. The result is always a permutation of
    ``range(len(lengths))`` — no row is dropped or duplicated — and is fully
    determined by ``seed``.
    """
    n = len(lengths)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    if factor < 1:
        factor = 1
    window = max(1, batch_size) * factor
    grouped: list[int] = []
    for start in range(0, n, window):
        chunk = order[start : start + window]
        chunk.sort(key=lambda idx: lengths[idx], reverse=True)
        grouped.extend(chunk)
    return grouped


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
    overlength_policy: Literal["error", "truncate"] = "error",
    loss_reduction: Literal["none", "sequence_mean", "token_mean"] = "none",
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
        # Qwen-family templates keep reasoning only after the last genuine user
        # query (tool observations do not reset the boundary). Apply the same
        # policy as serving before the cookbook renderer sees structured parts.
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

    if overlength_policy not in ("error", "truncate"):
        raise SFTConfigError(f"Unknown overlength policy {overlength_policy!r}; use 'error' or 'truncate'.")
    if loss_reduction not in ("none", "sequence_mean", "token_mean"):
        raise SFTConfigError(f"Unknown SFT loss reduction {loss_reduction!r}; use 'none', 'sequence_mean', or 'token_mean'.")
    if max_length is not None and model_input.length > max_length:
        if overlength_policy == "error":
            raise SFTConfigError(
                f"SFT row renders to {model_input.length} tokens > data.max_length={max_length}. "
                "The trajectory was not truncated. Drop it whole during preprocessing, raise "
                "max_length, or explicitly select overlength_policy='truncate' for the lossy "
                "compatibility behavior."
            )
        _warn_truncation(model_input.length, max_length)
    datum_max_length = max_length if overlength_policy == "truncate" else None
    reduction = "mean" if loss_reduction == "sequence_mean" else "none"
    return datum_from_model_input_weights(
        model_input,
        weights,
        datum_max_length,
        reduction=reduction,
    )


def count_loss_tokens(datums: list[tinker.Datum]) -> int:
    """Count supervised positions independently of loss-weight scaling."""
    return sum(1 for datum in datums for weight in datum.loss_fn_inputs["weights"].data if weight > 0)


def _normalize_token_mean(datums: list[tinker.Datum]) -> None:
    """Make one batch loss the mean over all of its supervised tokens."""
    total_weight = sum(float(weight) for datum in datums for weight in datum.loss_fn_inputs["weights"].data)
    if total_weight <= 0:
        raise SFTConfigError("SFT batch has no trainable tokens after rendering and masking.")
    for datum in datums:
        weights = datum.loss_fn_inputs["weights"]
        datum.loss_fn_inputs["weights"] = tinker.TensorData(
            data=[float(weight) / total_weight for weight in weights.data],
            dtype=weights.dtype,
            shape=list(weights.shape),
        )


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
        group_by_length: bool = False,
        length_group_factor: int = 50,
        overlength_policy: Literal["error", "truncate"] = "error",
        loss_reduction: Literal["none", "sequence_mean", "token_mean"] = "none",
    ):
        self.renderer = renderer
        if batch_size <= 0:
            raise SFTConfigError(f"SFT batch_size must be positive, got {batch_size}.")
        if max_length is not None and max_length <= 1:
            raise SFTConfigError(f"SFT max_length must be greater than 1, got {max_length}.")
        if overlength_policy not in ("error", "truncate"):
            raise SFTConfigError(f"Unknown overlength policy {overlength_policy!r}; use 'error' or 'truncate'.")
        if loss_reduction not in ("none", "sequence_mean", "token_mean"):
            raise SFTConfigError(f"Unknown SFT loss reduction {loss_reduction!r}; use 'none', 'sequence_mean', or 'token_mean'.")
        self.batch_size = batch_size
        self.max_length = max_length
        self.last_only = last_only
        self.group_by_length = group_by_length
        self.length_group_factor = max(1, int(length_group_factor))
        self.overlength_policy = overlength_policy
        self.loss_reduction = loss_reduction

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

        # Explicit per-epoch row ordering (identity until set_epoch); get_batch
        # reads rows through it, so shuffling never mutates the base dataset and
        # the length-grouping order can be recomputed each epoch. Row-length
        # proxies are precomputed once, only when grouping is on.
        self._order = list(range(len(self.dataset)))
        self._lengths: list[int] | None = None
        if self.group_by_length:
            # Per-row indexing (not column access): the rLLM Dataset object and a
            # parquet-backed HF dataset both index by int but only HF supports
            # string columns.
            self._lengths = [_message_byte_length(self.dataset[i]["messages"]) for i in range(len(self.dataset))]
            logger.info(f"Length-grouped batching enabled (factor={self.length_group_factor})")

        logger.info(f"Loaded {len(self.dataset)} examples from {source}")
        logger.info(f"Masking: CUSTOMIZED (derive last_only={last_only} for flag-less rows)")

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self._order))
        datums = []
        for pos in range(start_idx, end_idx):
            i = self._order[pos]
            row = self.dataset[i]
            try:
                datums.append(
                    conversation_to_datum(
                        row["messages"],
                        self.renderer,
                        self.max_length,
                        self.last_only,
                        tools=row.get("tools"),
                        overlength_policy=self.overlength_policy,
                        loss_reduction=self.loss_reduction,
                    )
                )
            except SFTConfigError as e:
                raise SFTConfigError(f"dataset row {i}: {e}") from e
        if self.loss_reduction == "token_mean":
            _normalize_token_mean(datums)
        elif datums and count_loss_tokens(datums) == 0:
            raise SFTConfigError("SFT batch has no trainable tokens after rendering and masking.")
        return datums

    def set_epoch(self, seed: int = 0):
        n = len(self.dataset)
        if self.group_by_length and self._lengths is not None:
            self._order = length_grouped_order(self._lengths, self.batch_size, self.length_group_factor, seed)
            logger.info(f"Length-grouped order for epoch (seed={seed}, {n} samples)")
        else:
            order = list(range(n))
            random.Random(seed).shuffle(order)
            self._order = order
            logger.info(f"Shuffled dataset with seed {seed} ({n} samples)")

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def data_cursor_for_step(self, completed_steps: int) -> int:
        """Raw-row cursor after ``completed_steps`` optimizer batches."""
        batches_per_epoch = len(self)
        if batches_per_epoch == 0:
            return 0
        completed_epochs, batches_in_epoch = divmod(completed_steps, batches_per_epoch)
        rows_per_epoch = len(self.dataset)
        return completed_epochs * rows_per_epoch + min(
            batches_in_epoch * self.batch_size,
            rows_per_epoch,
        )

    def step_for_data_cursor(self, data_consumed: int) -> int:
        """Inverse of :meth:`data_cursor_for_step` at batch boundaries."""
        rows_per_epoch = len(self.dataset)
        if rows_per_epoch == 0:
            return 0
        completed_epochs, rows_in_epoch = divmod(data_consumed, rows_per_epoch)
        batches_in_epoch = math.ceil(rows_in_epoch / self.batch_size)
        return completed_epochs * len(self) + batches_in_epoch


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
    group_by_length: bool = False,
    length_group_factor: int = 50,
    overlength_policy: Literal["error", "truncate"] = "error",
    loss_reduction: Literal["none", "sequence_mean", "token_mean"] = "none",
) -> tuple[TinkerSFTDataset, TinkerSFTDataset | None]:
    """Create train and optional validation datasets for Tinker SFT.

    ``group_by_length`` applies only to the train dataset: validation iterates
    every batch and reduces over all tokens, so its batch composition does not
    affect the metric — only the (single, sequential) pass cost, which the extra
    length-proxy computation would not repay.
    """
    if val_batch_size is None:
        val_batch_size = batch_size

    train_dataset = TinkerSFTDataset(
        dataset_or_files=train_data,
        renderer=renderer,
        batch_size=batch_size,
        max_length=max_length,
        last_only=last_only,
        max_samples=max_train_samples,
        group_by_length=group_by_length,
        length_group_factor=length_group_factor,
        overlength_policy=overlength_policy,
        loss_reduction=loss_reduction,
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
            overlength_policy=overlength_policy,
            # Validation is always reported as a token-weighted NLL.
            loss_reduction="none",
        )

    return train_dataset, val_dataset

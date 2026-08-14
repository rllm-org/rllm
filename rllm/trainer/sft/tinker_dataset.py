"""Tinker SFT dataset: converts rLLM ``messages`` rows into Tinker Datums.

Migrated from the (removed) ``rllm.trainer.deprecated.tinker_sft_dataset``.
Imported lazily by :class:`rllm.trainer.sft.tinker_backend.TinkerSFTBackend`, so
``tinker``/``tinker_cookbook`` are only required when actually training on tinker.
"""

from __future__ import annotations

import json
import logging

import datasets
import tinker
import torch
from tinker_cookbook.renderers import Message
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDataset

from rllm.data.sft_schema import SFTMessage, SFTSchemaError, normalize_messages
from rllm.renderers.types import Renderer
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


def _validate_rendered_attribution(rendered, message_count: int) -> None:
    """Require one valid source-message index per rendered token."""
    if len(rendered.message_indices) != len(rendered.token_ids):
        raise ValueError("renderer did not return one message attribution per token")
    invalid = next((index for index in rendered.message_indices if index < -1 or index >= message_count), None)
    if invalid is not None:
        raise ValueError(f"renderer returned invalid message index {invalid} for {message_count} messages")
    is_content = getattr(rendered, "is_content", None)
    if is_content and len(is_content) != len(rendered.token_ids):
        raise ValueError("renderer returned incomplete content-token metadata")


def _validate_trainable_targets_represented(
    renderer: Renderer,
    renderer_messages: list[dict],
    messages: list[SFTMessage],
    rendered,
    *,
    tools: list[dict] | None,
) -> None:
    """Fail when a full render rewrites an explicitly trainable target.

    Compare the full sequence through every earlier trainable target with that
    prefix rendered alone. This protects both the target and its causal context
    from history-dependent template rewrites.
    """
    for index, message in enumerate(messages):
        if not message.trainable or index == len(messages) - 1:
            continue
        prefix = renderer.render(renderer_messages[: index + 1], tools=tools)
        _validate_rendered_attribution(prefix, index + 1)
        if rendered.token_ids[: len(prefix.token_ids)] != prefix.token_ids:
            raise ValueError(
                f"the rendered prefix through trainable message {index} is rewritten when later turns are present. Explode this trajectory per target, or store that target as a separate SFT example."
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

    Normalizes through :mod:`rllm.data.sft_schema`, then renders with the same
    canonical :mod:`rllm.renderers` implementation used for inference. The
    renderer returns a source-message index for every token; ``trainable`` maps
    those indices to the loss mask. When the renderer also identifies message
    body tokens, template scaffolding is excluded. ``last_only`` only selects
    the default for source rows without a complete explicit mask.

    Schema/validation failures are re-raised as :class:`SFTConfigError` with the
    failing row's context.
    """
    default_trainable = "last" if last_only else "all"
    try:
        messages = normalize_messages(conversation, default_trainable=default_trainable)
        renderer_messages = [_to_renderer_message(message) for message in messages]
        renderer_tools = _validate_tools(tools)
        rendered = renderer.render(
            renderer_messages,
            tools=renderer_tools,
        )
        _validate_rendered_attribution(rendered, len(messages))
        _validate_trainable_targets_represented(
            renderer,
            renderer_messages,
            messages,
            rendered,
            tools=renderer_tools,
        )
    except SFTSchemaError as e:
        raise SFTConfigError(f"SFT row failed schema normalization: {e}\n  row={_row_context(conversation)}") from e
    except (TypeError, ValueError) as e:
        raise SFTConfigError(f"SFT row failed canonical rendering: {e}\n  row={_row_context(conversation)}") from e

    is_content = getattr(rendered, "is_content", None)
    content_mask = is_content or [True] * len(rendered.token_ids)
    weights = torch.tensor(
        [float(index >= 0 and bool(messages[index].trainable) and content) for index, content in zip(rendered.message_indices, content_mask, strict=True)],
        dtype=torch.float32,
    )
    model_input = tinker.ModelInput.from_ints(rendered.token_ids)
    if max_length is not None and model_input.length > max_length:
        _warn_truncation(model_input.length, max_length)
    return datum_from_model_input_weights(model_input, weights, max_length)


def _to_renderer_message(message: SFTMessage) -> dict:
    """Lower one canonical SFT message to the renderer/OpenAI message shape.

    Structured parts remain canonical in storage; the rendering contract has a
    single visible-text field and a sibling ``reasoning_content`` field. Joining
    parts by kind preserves their bytes and avoids provider-specific objects.
    """
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    seen_text = False
    for part in message.content:
        if part.type == "text":
            seen_text = True
            text_parts.append(part.text)
        elif seen_text:
            raise ValueError("thinking parts must precede visible text; the renderer schema cannot preserve interleaved text/thinking order")
        else:
            thinking_parts.append(part.thinking)

    output: dict = {
        "role": message.role,
        "content": "".join(text_parts),
        "trainable": bool(message.trainable),
    }
    thinking = "".join(thinking_parts)
    if thinking:
        output["reasoning_content"] = thinking
    if message.tool_calls:
        output["tool_calls"] = []
        for index, tool_call in enumerate(message.tool_calls):
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                raise ValueError(f"tool call {index} has invalid JSON arguments") from e
            if not isinstance(arguments, dict):
                raise ValueError(f"tool call {index} arguments must decode to an object")
            output["tool_calls"].append(tool_call.model_dump(exclude_none=True))
    if message.tool_call_id is not None:
        output["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        output["name"] = message.name
    return output


def _validate_tools(tools: list[dict] | None) -> list[dict] | None:
    """Validate declarations before a renderer can ignore malformed entries."""
    if not tools:
        return None
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError(f"tool declaration {index} must be an OpenAI function tool")
        function = tool.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            raise ValueError(f"tool declaration {index} needs a non-empty function.name")
        if not isinstance(function.get("parameters", {}), dict):
            raise ValueError(f"tool declaration {index} function.parameters must be an object")
    return tools


class TinkerSFTDataset(SupervisedDataset):
    """Dataset for Tinker SFT that loads from rLLM sources.

    Accepts a HuggingFace/rLLM Dataset object (from DatasetRegistry) or parquet
    file path(s) with a ``messages`` column, renders through rLLM's production
    renderer, and yields Tinker-compatible Datums in batches.
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

"""Canonical SFT row schema — the single normalization boundary for messages.

SFT datasets are ``{"messages": [...]}`` rows. Those messages arrive in several
shapes: legacy ``{"role", "content": str}`` turns, structured parts-list turns
with per-message ``trainable`` flags (``rllm dataset from-eval``'s automerge),
and parquet round-trip artifacts (pandas/pyarrow unify a per-row struct schema
and stamp ``None`` into keys only *some* messages carry — ``tool_calls``,
``trainable``, part cross-keys). This module normalizes all of them into a small
set of pydantic models, so the tinker loader can always render with tinker's
``CUSTOMIZED`` masking (the data alone decides the loss mask).

Pure pydantic — no ``tinker``/``tinker_cookbook`` import at module load. The only
tinker dependency is the lazy import inside :meth:`SFTMessage.to_tinker_message`,
which builds tinker's structured ``ToolCall`` objects for the renderer.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Message-level keys we recognize; everything else is dropped during cleanup.
_MESSAGE_KEYS = ("role", "content", "trainable", "tool_calls", "tool_call_id", "name")
# Optional keys that arrive as ``None`` from a parquet struct-unification round
# trip; a ``None`` here means "absent", so we drop the key entirely.
_OPTIONAL_NONE_KEYS = ("tool_calls", "tool_call_id", "name")


class SFTSchemaError(ValueError):
    """Raised when an SFT row/message cannot be normalized to the schema."""


# --- content parts -----------------------------------------------------------


class TextPart(BaseModel):
    """Visible text content in a message."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class ThinkingPart(BaseModel):
    """Model's internal reasoning (chain-of-thought) as a content part.

    Preserved model-agnostically: the renderer decides the reasoning format at
    training time (deepseek ``<think>``, qwen, harmony, ...).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["thinking"] = "thinking"
    thinking: str


# Discriminated on ``type`` so an unknown part type is a clean validation error.
ContentPart = Annotated[TextPart | ThinkingPart, Field(discriminator="type")]


# --- tool calls --------------------------------------------------------------


class ToolFunction(BaseModel):
    """OpenAI-format function body: a tool name and JSON-string arguments."""

    name: str
    arguments: str


class SFTToolCall(BaseModel):
    """Structured tool invocation (OpenAI/kosong ``tool_calls`` element)."""

    type: Literal["function"] = "function"
    id: str | None = None
    function: ToolFunction


# --- message + row -----------------------------------------------------------


class SFTMessage(BaseModel):
    """One conversation turn.

    ``content`` is always a parts list (str/None content is coerced upstream by
    :func:`normalize_message_dict`). ``trainable`` drives tinker's ``CUSTOMIZED``
    masking; :func:`normalize_messages` guarantees it is a real bool before the
    renderer sees it.
    """

    model_config = ConfigDict(extra="forbid")

    role: str
    content: list[ContentPart]
    trainable: bool | None = None
    tool_calls: list[SFTToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def text(self) -> str:
        """Concatenated visible text across the message's ``TextPart``s."""
        return "".join(p.text for p in self.content if isinstance(p, TextPart))

    def thinking(self) -> str:
        """Concatenated reasoning across the message's ``ThinkingPart``s."""
        return "".join(p.thinking for p in self.content if isinstance(p, ThinkingPart))

    def to_tinker_message(self) -> dict:
        """A plain dict shaped as tinker_cookbook's ``Message`` TypedDict.

        Content parts are re-emitted WITHOUT any ``None`` cross-keys; ``trainable``
        is always present as a bool (CUSTOMIZED requires it); ``tool_calls`` is a
        list of tinker ``ToolCall`` pydantic objects and is included ONLY when
        present (an absent key, never ``tool_calls=None``, so the renderer never
        iterates ``None``); ``tool_call_id``/``name`` appear only when set.
        """
        from tinker_cookbook.renderers import ToolCall  # lazy: tinker only needed here

        msg: dict = {
            "role": self.role,
            "content": [_part_to_dict(p) for p in self.content],
            "trainable": bool(self.trainable),
        }
        if self.tool_calls:
            msg["tool_calls"] = [ToolCall(id=tc.id, function=ToolCall.FunctionBody(name=tc.function.name, arguments=tc.function.arguments)) for tc in self.tool_calls]
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            msg["name"] = self.name
        return msg


class SFTRow(BaseModel):
    """One SFT dataset row: a ``messages`` list plus arbitrary extra columns.

    Extra row-level fields (``task_id``, ``reward``, ``source_run``, ...) are
    preserved so curation metadata survives a normalize round trip.
    """

    model_config = ConfigDict(extra="allow")

    messages: list[SFTMessage]

    def to_record(self) -> dict:
        """A plain-dict record (``model_dump(exclude_none=True)``); content stays
        a parts list and ``None`` optional fields are dropped."""
        return self.model_dump(exclude_none=True)


# --- normalization helpers ---------------------------------------------------


def _part_to_dict(part: ContentPart) -> dict:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ThinkingPart):
        return {"type": "thinking", "thinking": part.thinking}
    raise SFTSchemaError(f"unsupported content part: {part!r}")  # pragma: no cover


def _clean_part(part):
    """Drop ``None``-valued cross-keys from a content part dict.

    A parquet struct-unification round trip stamps every part-key onto every
    part; a ``TextPart`` then carries ``thinking=None`` (and vice versa). A
    ``None`` cross-key means "absent", so strip it before validation.
    """
    if isinstance(part, dict):
        return {k: v for k, v in part.items() if v is not None}
    return part


def normalize_message_dict(msg: dict) -> dict:
    """Pre-validation cleanup of a raw message dict (returns a plain dict).

    - Drops unknown message keys (e.g. ``weight``) and ``None``-valued
      ``tool_calls``/``tool_call_id``/``name`` (parquet round-trip artifacts).
    - Coerces ``str`` content to ``[{"type": "text", "text": ...}]`` and ``None``
      content to ``[{"type": "text", "text": ""}]``.
    - Strips ``None`` cross-keys from every content part.
    - Never touches ``trainable`` (its ``None``/absence drives derivation later).
    """
    if not isinstance(msg, dict):
        return msg  # let SFTMessage validation raise a clear error

    out: dict = {}
    for key in _MESSAGE_KEYS:
        if key not in msg:
            continue
        value = msg[key]
        if key in _OPTIONAL_NONE_KEYS and value is None:
            continue
        out[key] = value

    if "content" in out:
        content = out["content"]
        if content is None:
            out["content"] = [{"type": "text", "text": ""}]
        elif isinstance(content, str):
            out["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            out["content"] = [_clean_part(p) for p in content]
    return out


def normalize_messages(messages, default_trainable: str = "all") -> list[SFTMessage]:
    """Clean and validate a conversation into ``SFTMessage`` objects.

    Trainable-flag policy: if ANY message lacks a real bool ``trainable`` (missing
    or ``None``), the WHOLE conversation's flags are derived — assistant turns
    train (only the *last* assistant turn when ``default_trainable == "last"``),
    others do not — overriding any partial explicit flags. If every message
    already carries a real bool, the flags are kept untouched.
    """
    if not isinstance(messages, list) or len(messages) == 0:
        raise SFTSchemaError("'messages' must be a non-empty list of conversation turns.")

    # Provider-style reasoning siblings are source fields, not disposable
    # metadata. The messages bridge is the ingestion boundary that lifts them
    # into canonical thinking parts; reaching core normalization with one still
    # present means that boundary was skipped.
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        reasoning_key = next(
            (key for key in ("reasoning_content", "reasoning") if message.get(key) is not None),
            None,
        )
        if reasoning_key is not None:
            raise SFTSchemaError(f"message {i} contains source field {reasoning_key!r}; import it through `rllm dataset import --format messages` so reasoning is preserved as a thinking part.")

    cleaned = [normalize_message_dict(m) for m in messages]

    all_flagged = all(isinstance(m, dict) and isinstance(m.get("trainable"), bool) for m in cleaned)
    if not all_flagged:
        if default_trainable not in ("all", "last"):
            raise SFTSchemaError(f"unknown default_trainable {default_trainable!r}; use 'all' or 'last'.")
        last_assistant = max((i for i, m in enumerate(cleaned) if isinstance(m, dict) and m.get("role") == "assistant"), default=-1)
        for i, m in enumerate(cleaned):
            if not isinstance(m, dict):
                continue
            is_assistant = m.get("role") == "assistant"
            m["trainable"] = is_assistant and (default_trainable == "all" or i == last_assistant)

    out: list[SFTMessage] = []
    for i, m in enumerate(cleaned):
        try:
            out.append(SFTMessage.model_validate(m))
        except ValidationError as e:
            raise SFTSchemaError(f"message {i}: {e}") from e
    return out


def normalize_row(row: dict, default_trainable: str = "all") -> SFTRow:
    """Normalize a single ``{"messages": [...], **extra}`` row into an ``SFTRow``.

    Raises :class:`SFTSchemaError` for a non-dict row, a missing/empty
    ``messages`` field, or any message that fails validation. Extra row-level
    columns are preserved verbatim.
    """
    if not isinstance(row, dict):
        raise SFTSchemaError(f"row must be a dict with a 'messages' field, got {type(row).__name__}.")
    if "messages" not in row:
        raise SFTSchemaError("row is missing a 'messages' field.")

    messages = normalize_messages(row["messages"], default_trainable=default_trainable)
    extra = {k: v for k, v in row.items() if k != "messages"}
    try:
        return SFTRow(messages=messages, **extra)
    except ValidationError as e:  # pragma: no cover - extra="allow" makes this rare
        raise SFTSchemaError(str(e)) from e


def normalize_rows(rows, default_trainable: str = "all") -> list[SFTRow]:
    """Normalize a sequence of rows; errors name the failing row index."""
    out: list[SFTRow] = []
    for i, row in enumerate(rows):
        try:
            out.append(normalize_row(row, default_trainable=default_trainable))
        except SFTSchemaError as e:
            raise SFTSchemaError(f"row {i}: {e}") from e
    return out

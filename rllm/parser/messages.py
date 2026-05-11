"""rLLM-native message format.

This module is the canonical home for rLLM's message types and the
``MessageList`` / ``MessageSnapshot`` helpers used by agents and workflows.

Design choices (see ``tmp/chat_parser_refactor_loop/message_format.md`` for the
full discussion):

* **TypedDict, not dataclass.** rLLM messages are passed around the codebase
  as plain ``list[dict]`` (OpenAI Chat Completions shape) and that layout is
  load-bearing for every existing caller — tokenizers, JSON serialization,
  ``msg["key"]`` access. TypedDicts give us static typing without changing the
  runtime representation, so existing code is correctly-typed without any
  migration.

* **``reasoning`` is rLLM-canonical.** The HuggingFace / vLLM convention is
  ``reasoning_content``; rLLM uses ``reasoning`` and treats it as a first-class
  field on ``AssistantMessage``. ``from_openai`` normalizes incoming
  ``reasoning_content`` into ``reasoning`` so downstream rLLM code only ever
  has to look at one field.

* **Snapshots share dicts.** ``MessageSnapshot`` holds a reference to the
  underlying list plus a frozen length — no deep copy. ``to_list()`` returns
  a shallow copy of the slice; the message dicts themselves are shared with
  the live ``MessageList``. The contract is: **do not mutate message dicts
  after a snapshot has been taken from a ``MessageList`` they came from.**
  This is the same contract Python already enforces de facto via the
  ``Step.chat_completions`` deepcopy pattern, just made explicit and cheap.

* **When to use ``MessageList`` vs ``list[dict]``.** Use ``MessageList`` when
  you need O(1) snapshots for ``Step.chat_completions``-style turn capture in
  an agent. Use plain ``list[dict]`` everywhere else — every rLLM API that
  accepts messages accepts ``list[dict]`` at runtime.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired

logger = logging.getLogger(__name__)


# ── Content block types (for multimodal) ──────────────────────────────────


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrlBlock(TypedDict):
    type: Literal["image_url"]
    image_url: dict[str, str]


ContentBlock = TextBlock | ImageUrlBlock


# ── Tool call (OpenAI-compatible wire format) ─────────────────────────────


class FunctionCall(TypedDict):
    name: str
    arguments: str  # JSON string, OpenAI wire format


class ToolCallDict(TypedDict):
    id: str
    type: Literal["function"]
    function: FunctionCall


# ── Message types ─────────────────────────────────────────────────────────


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: str | list[ContentBlock] | None


class UserMessage(TypedDict):
    role: Literal["user"]
    content: str | list[ContentBlock] | None
    images: NotRequired[list[Any]]


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str | list[ContentBlock] | None
    reasoning: NotRequired[str | None]
    tool_calls: NotRequired[list[ToolCallDict]]


class ToolMessage(TypedDict):
    role: Literal["tool"]
    content: str | list[ContentBlock] | None
    tool_call_id: NotRequired[str]
    name: NotRequired[str]


Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage
Messages = list[Message]


# ── MessageList: append-only history with O(1) snapshots ──────────────────


class MessageList:
    """Append-only message history with O(1) snapshots.

    Wraps an internal list. Supports the subset of ``list`` operations that
    make sense for an append-only log: ``append``, ``extend``, iteration,
    indexing, length, and ``+ other`` (returns a plain list, for workflow
    compat). ``snapshot()`` returns a frozen view that shares the underlying
    storage.
    """

    __slots__ = ("_data",)

    def __init__(self, messages: list[Message] | None = None):
        self._data: list[Message] = list(messages) if messages else []

    def append(self, msg: Message) -> None:
        self._data.append(msg)

    def extend(self, msgs: list[Message]) -> None:
        self._data.extend(msgs)

    def snapshot(self) -> MessageSnapshot:
        """Return a frozen view of the current history. O(1), no copy."""
        return MessageSnapshot(self._data, len(self._data))

    def to_list(self) -> list[Message]:
        """Shallow copy of the current contents as a plain list[dict]."""
        return list(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __add__(self, other: list[Message]) -> list[Message]:
        return list(self._data) + list(other)

    def __radd__(self, other: list[Message]) -> list[Message]:
        return list(other) + list(self._data)

    def __repr__(self) -> str:
        return f"MessageList({self._data!r})"


class MessageSnapshot:
    """Frozen, read-only view of a ``MessageList`` at a point in time.

    Holds a reference to the underlying list plus a frozen length. Indexing
    and iteration respect ``self._length`` — appends made to the source
    ``MessageList`` after this snapshot was taken are invisible here.

    **Shallow share contract.** The dicts inside this snapshot are *the same
    objects* as those in the source ``MessageList``. If you mutate a dict
    in-place (e.g. ``msg["content"] = ...``), that change WILL be visible
    through this snapshot. Don't do that — treat snapshots and their source
    list as immutable past the snapshot point.
    """

    __slots__ = ("_data", "_length")

    def __init__(self, data: list[Message], length: int):
        self._data = data
        self._length = length

    def to_list(self) -> list[Message]:
        """Shallow copy of the slice as a plain list[dict]."""
        return self._data[: self._length]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._length)
            return self._data[start:stop:step]
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError("MessageSnapshot index out of range")
        return self._data[idx]

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for i in range(self._length):
            yield self._data[i]

    def __add__(self, other: list[Message]) -> list[Message]:
        return self.to_list() + list(other)

    def __radd__(self, other: list[Message]) -> list[Message]:
        return list(other) + self.to_list()

    def __repr__(self) -> str:
        return f"MessageSnapshot(length={self._length}, data={self._data[: self._length]!r})"


# ── Conversion helpers ────────────────────────────────────────────────────


def _convert_image_to_url_block(image: Any) -> ImageUrlBlock | None:
    """Best-effort PIL Image → ``image_url`` base64 data-URL block.

    Returns None if the image cannot be encoded (e.g. PIL not installed).
    Callers can decide whether to drop the image with a warning or raise.
    """
    try:
        import base64
        import io

        buf = io.BytesIO()
        # Default to PNG; PIL's save() requires a format if the image has no filename.
        fmt = getattr(image, "format", None) or "PNG"
        image.save(buf, format=fmt)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        mime = f"image/{fmt.lower()}"
        return ImageUrlBlock(
            type="image_url",
            image_url={"url": f"data:{mime};base64,{encoded}"},
        )
    except Exception as exc:
        logger.warning("Could not convert image to image_url block: %s", exc)
        return None


def to_openai(messages: list[Message] | MessageList | MessageSnapshot) -> list[dict]:
    """Convert rLLM messages to OpenAI Chat Completions wire format.

    Behavior on rLLM-specific fields:

    * ``reasoning`` is dropped. OpenAI Chat Completions has no equivalent
      field on the message; reasoning surfaces via a separate API mechanism.
    * ``images`` on a UserMessage is folded into ``content`` as
      ``image_url`` blocks where possible. Images that fail to encode are
      dropped with a warning.
    * ``tool_calls`` pass through unchanged (already OpenAI shape).
    """
    if isinstance(messages, MessageList | MessageSnapshot):
        seq: list[Message] = messages.to_list()
    else:
        seq = list(messages)

    out: list[dict] = []
    for msg in seq:
        new_msg: dict = {k: v for k, v in msg.items() if k != "reasoning"}

        if new_msg.get("role") == "user" and "images" in new_msg:
            images = new_msg.pop("images")
            image_blocks: list[ContentBlock] = []
            for img in images or []:
                block = _convert_image_to_url_block(img)
                if block is not None:
                    image_blocks.append(block)
            if image_blocks:
                content = new_msg.get("content")
                if content is None:
                    new_msg["content"] = image_blocks
                elif isinstance(content, str):
                    text_block: ContentBlock = TextBlock(type="text", text=content)
                    new_msg["content"] = [text_block, *image_blocks]
                elif isinstance(content, list):
                    new_msg["content"] = [*content, *image_blocks]
                else:
                    new_msg["content"] = image_blocks

        out.append(new_msg)
    return out


def from_openai(messages: list[dict]) -> Messages:
    """Validate and normalize OpenAI-shape dicts into rLLM ``Messages``.

    The "normalization" we do here is minimal and specific:

    * If a message carries ``reasoning_content`` (HF/vLLM convention) and no
      ``reasoning`` (rLLM convention), rename it to ``reasoning``. Downstream
      rLLM code reads only ``reasoning``.
    * Every message must have a string ``role``; anything else raises.
    * Unknown fields are preserved (TypedDicts are dicts at runtime; an
      unrecognized key is not a validation error).
    """
    out: Messages = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise TypeError(f"message[{i}] is not a dict: {type(msg).__name__}")
        role = msg.get("role")
        if not isinstance(role, str):
            raise ValueError(f"message[{i}] is missing a string 'role' field")

        normalized = dict(msg)
        if "reasoning_content" in normalized and "reasoning" not in normalized:
            normalized["reasoning"] = normalized.pop("reasoning_content")
        elif "reasoning_content" in normalized:
            # Both present — keep canonical, drop the alias.
            normalized.pop("reasoning_content")

        out.append(normalized)  # type: ignore[arg-type]
    return out


__all__ = [
    "AssistantMessage",
    "ContentBlock",
    "FunctionCall",
    "ImageUrlBlock",
    "Message",
    "MessageList",
    "MessageSnapshot",
    "Messages",
    "SystemMessage",
    "TextBlock",
    "ToolCallDict",
    "ToolMessage",
    "UserMessage",
    "from_openai",
    "to_openai",
]

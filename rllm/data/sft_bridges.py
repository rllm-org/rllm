"""Source-format bridges: raw dataset rows -> canonical :class:`SFTRow` objects.

A *bridge* is the ingestion boundary in front of the SFT schema
(:mod:`rllm.data.sft_schema`). Raw SFT data arrives in a couple of shapes, and
each bridge knows how to turn one of them into schema ``SFTRow`` objects that the
tinker loader can render with ``CUSTOMIZED`` masking:

- ``messages`` — plain OpenAI ``{"messages": [...]}`` rows. Delegates straight to
  :func:`rllm.data.sft_schema.normalize_rows`, deriving the ``trainable`` mask
  from ``train_on`` (``"all"`` assistant turns, or only the ``"last"`` one).
- ``think-tags`` — rows whose assistant turns carry a leading
  ``<think>...</think>`` block, a common convention for distilled reasoning
  traces (R1-style exports, many HF distill datasets). The chain-of-thought is
  split into a :class:`ThinkingPart`; every non-``messages`` top-level key is
  carried through verbatim as row-level metadata. By default the conversation is
  *exploded* into one row per assistant turn (history CoT stripped, single
  trainable target), which is the shape a next-token SFT loss wants.

Both bridges return ``list[SFTRow]``; callers persist ``[r.to_record() for r in
rows]``. Malformed rows raise :class:`SFTSchemaError` naming the failing row
index. Row metadata is passed through as-is; if you want canonical column names
(``task_id``, ``reward``, ...) rename the keys in your own bridge or preprocess
the file first.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from rllm.data.sft_schema import (
    SFTMessage,
    SFTRow,
    SFTSchemaError,
    normalize_message_dict,
    normalize_rows,
)

# Leading ``<think>COT</think>REST`` block, whitespace-tolerant, DOTALL so the
# CoT and the trailing content may both span newlines. Anchored at the start via
# ``match`` + ``\s*`` — a ``<think>`` that is not the leading token is left as
# plain text.
_THINK_RE = re.compile(r"\s*<think>(.*?)</think>(.*)", re.DOTALL)


# --- shared helpers ----------------------------------------------------------


def _content_to_parts(role: str, content: Any) -> list[dict]:
    """Turn a raw message ``content`` into a schema parts-list (list of dicts).

    An assistant ``str`` with a leading ``<think>...</think>`` block is split
    into a ``thinking`` part (CoT ``strip``ed) plus a ``text`` part (remainder
    left-stripped). Any other ``str`` becomes a single ``text`` part. ``None``
    and already-structured ``list`` content are coerced the same way the schema
    does, so this bridge tolerates partially-structured inputs too.
    """
    if isinstance(content, str):
        if role == "assistant":
            m = _THINK_RE.match(content)
            if m:
                return [
                    {"type": "thinking", "thinking": m.group(1).strip()},
                    {"type": "text", "text": m.group(2).lstrip()},
                ]
        return [{"type": "text", "text": content}]
    if content is None:
        return [{"type": "text", "text": ""}]
    if isinstance(content, list):
        # Reuse the schema's part cleanup (drops ``None`` cross-keys).
        return list(normalize_message_dict({"role": role, "content": content}).get("content", []))
    raise SFTSchemaError(f"unsupported content type {type(content).__name__} for role {role!r}.")


def _strip_thinking(parts: list[dict]) -> list[dict]:
    """Drop ``thinking`` parts — history turns keep only their visible text."""
    return [p for p in parts if not (isinstance(p, dict) and p.get("type") == "thinking")]


def _make_message(raw_msg: dict, parts: list[dict], trainable: bool, idx: int) -> SFTMessage:
    """Build a validated :class:`SFTMessage` from a raw turn + computed parts.

    Recognised non-content keys (``tool_calls``/``tool_call_id``/``name``) are
    carried through via the schema's own message cleanup; ``content`` and
    ``trainable`` are then overwritten with the bridge's decisions.
    """
    cleaned = normalize_message_dict(raw_msg)
    cleaned["content"] = parts
    cleaned["trainable"] = trainable
    try:
        return SFTMessage.model_validate(cleaned)
    except ValidationError as e:
        raise SFTSchemaError(f"message {idx}: {e}") from e


def _row_fields(row: dict) -> dict:
    """Carry every non-``messages`` top-level key through as row-level metadata.

    Keys are passed through verbatim — underscore-prefixed keys included, no
    renames and no silent drops. :class:`SFTRow` is ``extra=allow`` so these land
    unchanged in ``to_record()``. Callers who want canonical column names
    (``task_id``, ``reward``, ...) should rename keys before/after the bridge.
    """
    return {key: value for key, value in row.items() if key != "messages"}


def _bridge_think_row(row: dict, explode: bool) -> list[SFTRow]:
    """Bridge one think-tagged row into one (no-explode) or many (explode) rows."""
    if not isinstance(row, dict):
        raise SFTSchemaError(f"row must be a dict with a 'messages' field, got {type(row).__name__}.")
    if "messages" not in row:
        raise SFTSchemaError("row is missing a 'messages' field.")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        raise SFTSchemaError("'messages' must be a non-empty list of conversation turns.")
    for j, m in enumerate(messages):
        if not isinstance(m, dict):
            raise SFTSchemaError(f"message {j}: expected a dict, got {type(m).__name__}.")

    roles = [m.get("role") for m in messages]
    full_parts = [_content_to_parts(role, m.get("content")) for role, m in zip(roles, messages, strict=True)]
    hist_parts = [_strip_thinking(p) for p in full_parts]
    fields = _row_fields(row)

    if not explode:
        msgs = [_make_message(messages[j], full_parts[j], roles[j] == "assistant", j) for j in range(len(messages))]
        return [SFTRow(messages=msgs, **fields)]

    rows: list[SFTRow] = []
    for target in (j for j, role in enumerate(roles) if role == "assistant"):
        history = [_make_message(messages[j], hist_parts[j], False, j) for j in range(target)]
        history.append(_make_message(messages[target], full_parts[target], True, target))
        rows.append(SFTRow(messages=history, **fields))
    return rows


# --- bridges -----------------------------------------------------------------


def bridge_messages(rows: Sequence[dict], *, train_on: str = "all") -> list[SFTRow]:
    """Bridge plain OpenAI ``{"messages": [...]}`` rows via the schema.

    ``train_on`` picks the derived loss mask: ``"all"`` trains every assistant
    turn, ``"last"`` only the final one. Explicit per-message ``trainable`` flags
    (when *every* message carries one) are preserved by the schema.
    """
    return normalize_rows(rows, default_trainable=train_on)


def bridge_think_tags(rows: Sequence[dict], *, explode: bool = True) -> list[SFTRow]:
    """Bridge ``<think>``-tagged rows into schema rows.

    ``explode=True`` (default) emits one row per assistant turn: history turns
    are non-trainable with their CoT stripped, and the single final assistant
    turn is the trainable target (CoT kept). ``explode=False`` emits one row per
    conversation with every assistant turn trainable and CoT kept.
    """
    out: list[SFTRow] = []
    for i, row in enumerate(rows):
        try:
            out.extend(_bridge_think_row(row, explode))
        except SFTSchemaError as e:
            raise SFTSchemaError(f"row {i}: {e}") from e
    return out


# --- registry ----------------------------------------------------------------

BRIDGES: dict[str, Callable[..., list[SFTRow]]] = {
    "messages": bridge_messages,
    "think-tags": bridge_think_tags,
}


def get_bridge(name: str) -> Callable[..., list[SFTRow]]:
    """Look up a bridge by source-format name, or raise listing valid names."""
    try:
        return BRIDGES[name]
    except KeyError:
        raise SFTSchemaError(f"unknown source format {name!r}; valid formats: {', '.join(sorted(BRIDGES))}.") from None

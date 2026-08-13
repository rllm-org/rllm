"""Source-format bridges: raw dataset rows -> canonical :class:`SFTRow` objects.

A *bridge* is the ingestion boundary in front of the SFT schema
(:mod:`rllm.data.sft_schema`). Raw SFT data arrives in a couple of shapes, and
each bridge knows how to turn one of them into schema ``SFTRow`` objects that the
tinker loader can render with ``CUSTOMIZED`` masking:

- ``messages`` — plain OpenAI ``{"messages": [...]}`` rows, including the
  reasoning-model flavour where chain-of-thought rides in a sibling
  ``reasoning_content`` field. Derives the ``trainable`` mask from ``train_on``
  (``"all"`` assistant turns, or only the ``"last"`` one).
- ``think-tags`` — rows whose assistant turns carry a leading
  ``<think>...</think>`` block, a common convention for distilled reasoning
  traces (R1-style exports, many HF distill datasets). The chain-of-thought is
  split into a :class:`ThinkingPart`; every non-``messages`` top-level key is
  carried through verbatim as row-level metadata. By default the conversation is
  *exploded* into one row per assistant turn (history CoT stripped, single
  trainable target), which is the shape a next-token SFT loss wants.

Source shape varies far more than the schema does, so the *bridge* is where that
variance is absorbed — once, at dataset-build time — rather than in renderers and
trainers that would otherwise each have to cope with every export's quirks. Both
bridges therefore apply the same source-shape normalization before validation
(``reasoning_content`` lifted to a thinking part, JSON-string ``tool_calls``
decoded, already-parsed ``arguments`` re-encoded). Source text is preserved
verbatim; model-aware normalization belongs with the resolved renderer.

Both bridges return ``list[SFTRow]``; callers persist ``[r.to_record() for r in
rows]``. Malformed rows raise :class:`SFTSchemaError` naming the failing row
index. Row metadata is passed through as-is; if you want canonical column names
(``task_id``, ``reward``, ...) rename the keys in your own bridge or preprocess
the file first.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from rllm.data.sft_schema import (
    SFTMessage,
    SFTRow,
    SFTSchemaError,
    _has_structural_inline_thinking,
    normalize_message_dict,
    normalize_rows,
)

# Leading ``<think>COT</think>REST`` block, whitespace-tolerant, DOTALL so the
# CoT and the trailing content may both span newlines. Anchored at the start via
# ``match`` + ``\s*`` — a ``<think>`` that is not the leading token is left as
# plain text.
_THINK_RE = re.compile(r"\s*<think>(.*?)</think>(.*)", re.DOTALL)

# Sibling fields carrying chain-of-thought next to ``content`` in the
# OpenAI-compatible reasoning-model wire shape, in precedence order.
_REASONING_KEYS = ("reasoning_content", "reasoning")
_PART_PAYLOAD = {"text": "text", "thinking": "thinking"}


# --- source-shape normalization ----------------------------------------------


def _normalize_tool_call(call: Any) -> Any:
    """Canonicalize one ``tool_calls`` element to the OpenAI wire shape.

    ``arguments`` is a JSON *string* in the canonical schema, but exports that
    round-trip through a JSON/parquet writer often store it already parsed. Both
    shapes describe the same call, so re-encode the parsed one rather than
    rejecting the row.
    """
    if not isinstance(call, dict):
        return call
    fn = call.get("function")
    if not isinstance(fn, dict) or not isinstance(fn.get("arguments"), dict | list):
        return call
    try:
        arguments = json.dumps(fn["arguments"], ensure_ascii=False)
    except TypeError as e:
        raise SFTSchemaError(f"tool call arguments are not JSON-serializable: {e}") from e
    return {**call, "function": {**fn, "arguments": arguments}}


def _normalize_source_message(msg: Any) -> Any:
    """Coerce provider reasoning and tool-call fields into schema shapes."""
    if not isinstance(msg, dict):
        return msg
    out = dict(msg)

    tool_calls = out.get("tool_calls")
    if isinstance(tool_calls, str):
        if not tool_calls.strip():
            out.pop("tool_calls")
        else:
            try:
                tool_calls = json.loads(tool_calls)
            except ValueError as e:
                raise SFTSchemaError(f"tool_calls is a string but not valid JSON: {e}") from e
            if not isinstance(tool_calls, list):
                raise SFTSchemaError(f"tool_calls JSON must decode to a list, got {type(tool_calls).__name__}.")
            out["tool_calls"] = tool_calls
    if isinstance(out.get("tool_calls"), list):
        out["tool_calls"] = [_normalize_tool_call(call) for call in out["tool_calls"]]

    reasoning_values: list[tuple[str, str]] = []
    for key in _REASONING_KEYS:
        value = out.pop(key, None)
        if value is None or (isinstance(value, str) and value == ""):
            continue
        if not isinstance(value, str):
            raise SFTSchemaError(f"{key} must be a string, got {type(value).__name__}.")
        if out.get("role") != "assistant":
            raise SFTSchemaError(f"{key} is supported only on assistant messages, got role {out.get('role')!r}.")
        reasoning_values.append((key, value))
    if len({value for _, value in reasoning_values}) > 1:
        fields = ", ".join(key for key, _ in reasoning_values)
        raise SFTSchemaError(f"conflicting reasoning aliases ({fields}) carry different values.")

    reasoning = reasoning_values[0][1] if reasoning_values else ""
    if reasoning:
        content = out.get("content")
        if _has_structural_inline_thinking(content):
            fields = ", ".join(key for key, _ in reasoning_values)
            raise SFTSchemaError(f"sibling reasoning field ({fields}) conflicts with structural inline <think>...</think> content; choose one reasoning representation.")
        parts: list = [{"type": "thinking", "thinking": reasoning}]
        if isinstance(content, str):
            if content:
                parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            parts.extend(normalize_message_dict({"role": out.get("role"), "content": content}).get("content", []))
        elif content is not None:
            raise SFTSchemaError(f"assistant content must be a string, list, or null when reasoning is present, got {type(content).__name__}.")
        out["content"] = parts
    return out


def _payload_of(part: Any, kind: str | None = None) -> str | None:
    if not isinstance(part, dict):
        return None
    part_type = part.get("type")
    if kind is not None and part_type != kind:
        return None
    key = _PART_PAYLOAD.get(part_type) if isinstance(part_type, str) else None
    value = part.get(key) if key else None
    return value if isinstance(value, str) else None


# --- shared helpers ----------------------------------------------------------


def _split_think_tag(text: str) -> list[dict]:
    match = _THINK_RE.match(text)
    if not match:
        return [{"type": "text", "text": text}]
    return [
        {"type": "thinking", "thinking": match.group(1).strip()},
        {"type": "text", "text": match.group(2).lstrip()},
    ]


def _content_to_parts(role: str, content: Any) -> list[dict]:
    """Turn a raw message ``content`` into a schema parts-list (list of dicts).

    An assistant ``str`` with a leading ``<think>...</think>`` block is split
    into a ``thinking`` part (CoT ``strip``ed) plus a ``text`` part (remainder
    left-stripped). Any other ``str`` becomes a single ``text`` part. ``None``
    and already-structured ``list`` content are coerced the same way the schema
    does, so this bridge tolerates partially-structured inputs too.
    """
    if isinstance(content, str):
        return _split_think_tag(content) if role == "assistant" else [{"type": "text", "text": content}]
    if content is None:
        return [{"type": "text", "text": ""}]
    if isinstance(content, list):
        parts = list(normalize_message_dict({"role": role, "content": content}).get("content", []))
        if role != "assistant":
            return parts
        for index, part in enumerate(parts):
            if _payload_of(part, "text") is None:
                continue
            split = _split_think_tag(part["text"])
            return parts[:index] + split + parts[index + 1 :] if len(split) > 1 else parts
        return parts
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

    messages = [_normalize_source_message(m) for m in messages]
    roles = [m.get("role") for m in messages]
    full_parts = [_content_to_parts(role, m.get("content")) for role, m in zip(roles, messages, strict=True)]
    hist_parts = [_strip_thinking(p) for p in full_parts]
    fields = _row_fields(row)
    all_flagged = all(isinstance(m.get("trainable"), bool) for m in messages)

    if not explode:
        msgs = [
            _make_message(
                messages[j],
                full_parts[j],
                bool(messages[j]["trainable"]) if all_flagged else roles[j] == "assistant",
                j,
            )
            for j in range(len(messages))
        ]
        return [SFTRow(messages=msgs, **fields)]

    def _is_target(index: int) -> bool:
        if roles[index] != "assistant":
            return False
        return bool(messages[index].get("trainable")) if all_flagged else True

    rows: list[SFTRow] = []
    for target in (j for j in range(len(messages)) if _is_target(j)):
        history = [_make_message(messages[j], hist_parts[j], False, j) for j in range(target)]
        history.append(_make_message(messages[target], full_parts[target], True, target))
        rows.append(SFTRow(messages=history, **fields))
    return rows


# --- bridges -----------------------------------------------------------------


def bridge_messages(rows: Sequence[dict], *, train_on: str = "all") -> list[SFTRow]:
    """Bridge plain OpenAI ``{"messages": [...]}`` rows via the schema.

    ``train_on`` picks the derived loss mask: ``"all"`` trains every assistant
    turn, ``"last"`` only the final one. Explicit per-message ``trainable`` flags
    (when *every* message carries one) are preserved by the schema. Source-shape
    normalization (reasoning lifting and tool-call decoding) always applies.
    """
    prepared: list = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
            prepared.append(row)
            continue
        try:
            prepared.append({**row, "messages": [_normalize_source_message(m) for m in row["messages"]]})
        except SFTSchemaError as e:
            raise SFTSchemaError(f"row {i}: {e}") from e
    return normalize_rows(prepared, default_trainable=train_on)


def bridge_think_tags(rows: Sequence[dict], *, explode: bool = True) -> list[SFTRow]:
    """Bridge ``<think>``-tagged rows into schema rows.

    ``explode=True`` (default) emits one row per assistant target: history turns
    are non-trainable with their CoT stripped, and the final assistant turn is
    the trainable target. A complete source ``trainable`` mask selects targets;
    otherwise every assistant turn is selected. ``explode=False`` preserves a
    complete source mask and derives assistant targets when the mask is partial.
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

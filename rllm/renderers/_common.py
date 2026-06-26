"""Backend-agnostic normalization helpers shared by the renderer backends."""

from __future__ import annotations

import json
from typing import Any

from rllm.renderers.types import ToolSpec
from rllm.tools.tool_base import ToolCall


def iter_tool_specs(
    tools: list[ToolSpec] | list[dict[str, Any]] | None,
) -> list[ToolSpec]:
    """Normalize a heterogeneous ``tools`` argument to a list of :class:`ToolSpec`.

    Accepts native ``ToolSpec`` objects, OpenAI-style ``{"type": "function",
    "function": {...}}`` dicts, and bare ``{"name", "description", "parameters"}``
    dicts. Non-function tool dicts are skipped (mirrors
    ``TinkerEngine._build_messages_with_tools``).
    """
    if not tools:
        return []
    out: list[ToolSpec] = []
    for tool in tools:
        if isinstance(tool, ToolSpec):
            out.append(tool)
            continue
        if not isinstance(tool, dict):
            raise TypeError(f"Unrecognized tool spec type: {type(tool)}")
        if "function" in tool:
            if tool.get("type", "function") != "function":
                continue
            func = tool["function"]
        elif "name" in tool:
            func = tool
        else:
            # Not a recognizable function tool (e.g. {"type": "web_search"}); skip.
            continue
        out.append(
            ToolSpec(
                name=func["name"],
                description=func.get("description", ""),
                parameters=func.get("parameters", {}),
            )
        )
    return out


def normalize_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    """Convert backend tool-call objects to rLLM :class:`ToolCall` (name/arguments).

    Handles prime-rl / tinker-cookbook ``ToolCall(function=FunctionBody(...))``,
    flat ``ToolCall(name, arguments)``, and dict forms. Mirrors the conversion in
    ``TinkerEngine._parse_tinker_message``.
    """
    if not raw_tool_calls:
        return []
    calls: list[ToolCall] = []
    for tc in raw_tool_calls:
        if isinstance(tc, ToolCall):
            calls.append(tc)
        elif hasattr(tc, "function"):
            args = tc.function.arguments
            calls.append(
                ToolCall(
                    name=tc.function.name,
                    arguments=json.loads(args) if isinstance(args, str) else args,
                )
            )
        elif hasattr(tc, "name") and hasattr(tc, "arguments"):
            args = tc.arguments
            calls.append(
                ToolCall(
                    name=tc.name,
                    arguments=json.loads(args) if isinstance(args, str) else (args or {}),
                )
            )
        elif isinstance(tc, dict):
            args = tc.get("arguments", {})
            calls.append(
                ToolCall(
                    name=tc.get("name", ""),
                    arguments=json.loads(args) if isinstance(args, str) else (args or {}),
                )
            )
        else:
            raise TypeError(f"Unrecognized tool_call type: {type(tc)}")
    return calls

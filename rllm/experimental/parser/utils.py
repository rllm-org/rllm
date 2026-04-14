from __future__ import annotations

import json


def normalize_tools(tools: list) -> list[dict]:
    """Normalize tools to OpenAI-schema dicts for apply_chat_template.

    Accepts rllm Tool instances, dicts, JSON strings, or any pydantic model
    exposing model_dump().
    """
    from rllm.tools.tool_base import Tool

    out: list[dict] = []
    for tool in tools:
        if isinstance(tool, Tool):
            out.append(tool.json)
        elif isinstance(tool, dict):
            out.append(tool)
        elif isinstance(tool, str):
            out.append(json.loads(tool))
        elif hasattr(tool, "model_dump"):
            out.append(tool.model_dump(exclude_none=True))
        else:
            raise TypeError(f"Unsupported tool type: {type(tool)}")
    return out

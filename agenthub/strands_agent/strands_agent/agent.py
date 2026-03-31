"""Strands Agents framework ReAct agent plugin for rLLM.

Uses Strands' ``Agent`` with an ``OpenAIModel`` routed through the eval
proxy.  Adapts to any benchmark via TaskSpec.

Supports both text and multimodal (VLM) benchmarks by converting image
content blocks to Strands ``ContentBlock`` objects.
"""

from __future__ import annotations

import base64
from typing import Any

from strands import Agent
from strands.models.openai import OpenAIModel
from strands.types.content import ContentBlock
from strands.types.media import ImageContent, ImageSource

from rllm.experimental.eval.types import AgentConfig
from rllm.sdk.integrations.strands import RLLMTrajectoryHookProvider
from rllm.types import Episode


def _get_spec_and_data(task: Any) -> tuple[Any, dict]:
    """Extract TaskSpec and raw data from a task (Task object or dict)."""
    spec = None
    data = task
    if hasattr(task, "spec"):
        spec = task.spec
    if hasattr(task, "data"):
        data = task.data
    if isinstance(data, dict):
        return spec, data
    return spec, {"question": str(data)}


def _image_url_to_content_block(image_url: Any) -> ContentBlock | None:
    """Decode an OpenAI ``image_url`` dict to a Strands image ContentBlock."""
    if isinstance(image_url, dict):
        url = image_url.get("url", "")
    elif isinstance(image_url, str):
        url = image_url
    else:
        return None

    if not url or not url.startswith("data:"):
        return None

    try:
        header, encoded = url.split(",", 1)
        # header looks like "data:image/png;base64"
        mime = header.split(":")[1].split(";")[0]  # "image/png"
        fmt = mime.split("/")[1]  # "png"
        if fmt == "jpg":
            fmt = "jpeg"
        raw_bytes = base64.b64decode(encoded)
        return ContentBlock(
            image=ImageContent(
                format=fmt,
                source=ImageSource(bytes=raw_bytes),
            )
        )
    except Exception:
        return None


def _bridge_tool(tool_def: dict) -> dict:
    """Bridge an OpenAI tool dict (with ``_execute``) to a Strands tool config.

    Returns a dict suitable for Strands' tool registry with ``toolSpec``
    and ``execute`` keys.
    """
    func_spec = tool_def.get("function", {})
    name = func_spec.get("name", "tool")
    description = func_spec.get("description", "A tool.")
    parameters = func_spec.get("parameters", {})
    executor = tool_def.get("_execute")

    # Build Strands toolSpec
    input_schema = {"type": "object", "properties": {}, "required": []}
    props = parameters.get("properties", {})
    required = parameters.get("required", [])
    for param_name, param_spec in props.items():
        input_schema["properties"][param_name] = {
            "type": param_spec.get("type", "string"),
            "description": param_spec.get("description", ""),
        }
    input_schema["required"] = required

    def tool_handler(**kwargs):
        if executor is not None:
            return str(executor(**kwargs))
        return ""

    tool_handler.__name__ = name
    tool_handler.__doc__ = description
    tool_handler._tool_spec = {
        "name": name,
        "description": description,
        "inputSchema": {"json": input_schema},
    }

    return tool_handler


class StrandsAgentFlow:
    """Strands ReAct agent that works with any benchmark via TaskSpec."""

    def run(self, task: dict, config: AgentConfig) -> Episode:
        spec, data = _get_spec_and_data(task)

        system_prompt = "You are a capable AI assistant.\n\n"
        system_prompt += spec.instruction if spec else "Solve the given task."

        if spec:
            user_content = spec.render_input(data)
        else:
            user_content = data.get("question", str(data))

        # Convert multimodal content blocks to Strands ContentBlock list.
        # Strands accepts ``prompt`` as a string or list[ContentBlock].
        prompt: str | list[ContentBlock]
        if isinstance(user_content, list):
            content_blocks: list[ContentBlock] = []
            text_for_input = ""
            for block in user_content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = block.get("text", "")
                    content_blocks.append(ContentBlock(text=text))
                    if not text_for_input:
                        text_for_input = text
                elif block.get("type") == "image_url":
                    cb = _image_url_to_content_block(block.get("image_url", {}))
                    if cb is not None:
                        content_blocks.append(cb)
            prompt = content_blocks if content_blocks else str(user_content)
        elif isinstance(user_content, str):
            prompt = user_content
        else:
            prompt = str(user_content)

        # Create model routed through the eval proxy
        model = OpenAIModel(
            client_args={"base_url": config.base_url, "api_key": "EMPTY"},
            model_id=config.model,
        )

        # Set up tracing
        hook_provider = RLLMTrajectoryHookProvider()

        # Bridge tools from config metadata
        tools_meta: list[dict] = config.metadata.get("tools", [])
        bridged_tools = [_bridge_tool(t) for t in tools_meta if "function" in t]

        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=bridged_tools if bridged_tools else [],
            hooks=[hook_provider],
            callback_handler=None,
        )

        result = agent(prompt)
        answer = str(result) if result is not None else ""

        traj = hook_provider.get_trajectory()

        return Episode(
            task=data,
            trajectories=[traj],
            artifacts={"answer": answer},
        )


strands_agent = StrandsAgentFlow()

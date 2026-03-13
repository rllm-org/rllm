"""SmolAgents framework ReAct agent plugin for rLLM.

Uses SmolAgents' ``ToolCallingAgent`` with an ``OpenAIServerModel`` routed
through the eval proxy.  Adapts to any benchmark via TaskSpec.

Supports both text and multimodal (VLM) benchmarks by extracting images
from content blocks and passing them via SmolAgents' ``images`` parameter.
"""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image
from smolagents import OpenAIServerModel, Tool, ToolCallingAgent

from rllm.experimental.eval.types import AgentConfig
from rllm.sdk.integrations.smolagents import RLLMSmolAgentsTracer
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


def _data_url_to_pil(image_url: Any) -> Image.Image | None:
    """Decode a data-URI ``image_url`` dict to a PIL Image."""
    if isinstance(image_url, dict):
        url = image_url.get("url", "")
    elif isinstance(image_url, str):
        url = image_url
    else:
        return None

    if not url:
        return None

    try:
        if url.startswith("data:"):
            # data:image/png;base64,<b64>
            _, encoded = url.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(encoded)))
        # Could be a file path or HTTP URL — skip for now
        return None
    except Exception:
        return None


def _bridge_tool(tool_def: dict) -> Tool:
    """Bridge an OpenAI tool dict (with ``_execute``) to a SmolAgents ``Tool``."""
    func_spec = tool_def.get("function", {})
    name = func_spec.get("name", "tool")
    description = func_spec.get("description", "A tool.")
    parameters = func_spec.get("parameters", {})
    executor = tool_def.get("_execute")

    # Build inputs dict for SmolAgents (name -> {type, description})
    inputs: dict[str, dict[str, str]] = {}
    props = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    for param_name, param_spec in props.items():
        param_type = param_spec.get("type", "string")
        param_desc = param_spec.get("description", "")
        nullable = param_name not in required
        inputs[param_name] = {
            "type": param_type,
            "description": param_desc,
            "nullable": str(nullable).lower(),
        }

    # Dynamically create a Tool subclass
    attrs: dict[str, Any] = {
        "name": name,
        "description": description,
        "inputs": inputs,
        "output_type": "string",
    }

    def forward(self, **kwargs):
        if executor is not None:
            result = executor(**kwargs)
            return str(result)
        return ""

    attrs["forward"] = forward

    tool_cls = type(f"BridgedTool_{name}", (Tool,), attrs)
    return tool_cls()


class SmolAgentsFlow:
    """SmolAgents ReAct agent that works with any benchmark via TaskSpec."""

    def run(self, task: dict, config: AgentConfig) -> Episode:
        spec, data = _get_spec_and_data(task)

        instructions = "You are a capable AI assistant.\n\n"
        instructions += spec.instruction if spec else "Solve the given task."

        if spec:
            user_content = spec.render_input(data)
        else:
            user_content = data.get("question", str(data))

        # Separate text and images from multimodal content blocks.
        # SmolAgents expects text as the task string and PIL images via
        # the ``images`` parameter.
        images: list[Image.Image] = []
        if isinstance(user_content, list):
            text_parts = []
            for block in user_content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    img = _data_url_to_pil(block.get("image_url", {}))
                    if img is not None:
                        images.append(img)
            user_content = "\n".join(text_parts) if text_parts else str(user_content)
        elif not isinstance(user_content, str):
            user_content = str(user_content)

        # Create model routed through the eval proxy
        model = OpenAIServerModel(
            model_id=config.model,
            api_base=config.base_url,
            api_key="EMPTY",
        )

        # Wrap model for tracing
        tracer = RLLMSmolAgentsTracer()
        tracer._user_input = {"message": user_content[:500]}
        wrapped_model = tracer.wrap_model(model)

        # Bridge tools from config metadata
        tools_meta: list[dict] = config.metadata.get("tools", [])
        bridged_tools = [_bridge_tool(t) for t in tools_meta if "function" in t]

        agent = ToolCallingAgent(
            tools=bridged_tools,
            model=wrapped_model,
            instructions=instructions,
        )

        result = agent.run(user_content, images=images if images else None)
        answer = str(result) if result is not None else ""

        # Update tracer output
        tracer._last_output = answer
        tracer._trajectory_built = False
        traj = tracer.get_trajectory()

        return Episode(
            task=data,
            trajectories=[traj],
            artifacts={"answer": answer},
        )


smolagents_agent = SmolAgentsFlow()

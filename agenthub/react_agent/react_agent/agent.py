"""General-purpose ReAct agent using OpenAI Agents SDK via LiteLLM proxy.

Adapts to any benchmark via TaskSpec: uses ``spec.instruction`` for the system
prompt and ``spec.render_input(task)`` for the user message.  Routes all LLM
calls through the eval proxy using the Chat Completions API, which makes it
compatible with any provider supported by LiteLLM.

Supports both text and multimodal (VLM) benchmarks by converting image content
blocks to the Agents SDK's Responses API input format.
"""

from __future__ import annotations

from typing import Any

from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from agents.run import RunConfig

from rllm.experimental.eval.types import AgentConfig
from rllm.sdk.integrations.openai_agents import RLLMTrajectoryHooks
from rllm.types import Episode


def _to_agent_input(rendered: str | list[dict]) -> str | list[dict[str, Any]]:
    """Convert TaskSpec rendered input to Agents SDK input format.

    TaskSpec's render_input returns either a plain string (text tasks) or a
    list of Chat Completions content blocks (multimodal tasks).  The Agents
    SDK expects either a plain string or a list of Responses API input items.

    Chat Completions format (from TaskSpec):
        [{"type": "image_url", "image_url": {"url": "data:..."}},
         {"type": "text", "text": "question"}]

    Responses API format (for Agents SDK):
        [{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:...", "detail": "auto"},
            {"type": "input_text", "text": "question"}
        ]}]
    """
    if isinstance(rendered, str):
        return rendered

    # Convert Chat Completions content blocks → Responses API content blocks
    content: list[dict[str, Any]] = []
    for block in rendered:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "image_url":
            url = block.get("image_url", {})
            if isinstance(url, dict):
                url = url.get("url", "")
            content.append(
                {
                    "type": "input_image",
                    "image_url": url,
                    "detail": "auto",
                }
            )
        elif block.get("type") == "text":
            content.append(
                {
                    "type": "input_text",
                    "text": block.get("text", ""),
                }
            )

    if not content:
        # Fallback: extract text parts
        text_parts = [b.get("text", "") for b in rendered if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(text_parts) if text_parts else str(rendered)

    return [{"role": "user", "content": content}]


class ReactAgentFlow:
    """General-purpose ReAct agent that works with any benchmark via TaskSpec."""

    def run(self, task: dict, config: AgentConfig) -> Episode:
        spec = config.metadata.get("task_spec")

        # TaskSpec → instructions + user input
        instructions = "You are a capable AI assistant.\n\n"
        instructions += spec.instruction if spec else "Solve the given task."

        if spec:
            user_input = _to_agent_input(spec.render_input(task))
        else:
            user_input = task.get("question", str(task))

        agent = Agent(
            name="react",
            instructions=instructions,
        )

        # Route through the eval LiteLLM proxy via Chat Completions API,
        # per https://docs.litellm.ai/docs/tutorials/openai_agents_sdk
        model = LitellmModel(
            model=config.model,
            base_url=config.base_url,
            api_key="EMPTY",
        )

        # Run with trajectory hooks.
        # Disable Agents SDK's own tracing (we capture via RLLMTrajectoryHooks
        # instead), which also avoids the "OPENAI_API_KEY is not set" warning.
        hooks = RLLMTrajectoryHooks()
        result = Runner.run_sync(
            agent,
            user_input,
            hooks=hooks,
            run_config=RunConfig(model=model, tracing_disabled=True),
        )

        answer = result.final_output or ""
        traj = hooks.get_trajectory()

        return Episode(
            task=task,
            trajectories=[traj],
            artifacts={"answer": answer},
        )


react_agent = ReactAgentFlow()

"""ToolCallingMixin: reusable multi-turn tool-calling loop for sandboxed agents.

Tools are objects with ``.schema`` (OpenAI function tool definition) and
``.execute(sandbox, **args)`` method.
"""

from __future__ import annotations

import json
import logging

import openai

from rllm.sdk.sandbox.protocol import Sandbox
from rllm.types import Step

logger = logging.getLogger(__name__)


class ToolCallingMixin:
    """Reusable multi-turn tool-calling loop for sandboxed agents.

    Usage::

        class MyAgent(SandboxedAgentFlow, ToolCallingMixin):
            def run(self, task, config):
                steps, messages, final = self.run_tool_loop(
                    client, config.model, messages,
                    tools=[BashTool()], sandbox=self.sandbox,
                )
    """

    def run_tool_loop(
        self,
        client: openai.OpenAI,
        model: str,
        messages: list[dict],
        tools: list,
        sandbox: Sandbox,
        max_turns: int = 30,
        temperature: float = 0.0,
    ) -> tuple[list[Step], list[dict], str | None]:
        """Run multi-turn tool-calling loop.

        Args:
            client: OpenAI-compatible client.
            model: Model name.
            messages: Conversation history (mutated in place).
            tools: List of tool objects with ``.schema`` and ``.execute()``.
            sandbox: Sandbox instance to execute tools in.
            max_turns: Maximum number of LLM calls.
            temperature: Sampling temperature.

        Returns:
            Tuple of (steps, messages, final_content).
            ``final_content`` is the last assistant message without tool calls,
            or ``None`` if the loop hit max_turns.
        """
        tool_schemas = [t.schema for t in tools]
        tool_map = {t.schema["function"]["name"]: t for t in tools}
        steps: list[Step] = []

        for turn in range(max_turns):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_schemas if tool_schemas else openai.NOT_GIVEN,
                temperature=temperature,
            )
            choice = response.choices[0]
            msg = choice.message

            # Append assistant message to conversation
            messages.append(msg.model_dump(exclude_none=True))

            # No tool calls — agent is done
            if not msg.tool_calls:
                steps.append(
                    Step(
                        input=f"turn_{turn}",
                        output=msg.content or "",
                        done=True,
                    )
                )
                return steps, messages, msg.content

            # Process each tool call
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                tool = tool_map.get(fn_name)

                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"command": tc.function.arguments}

                if tool is None:
                    output = f"Unknown tool: {fn_name}"
                else:
                    try:
                        output = tool.execute(sandbox, **args)
                    except Exception as e:
                        logger.warning("Tool %s error: %s", fn_name, e)
                        output = f"Error executing {fn_name}: {e}"

                steps.append(Step(input=tc.function.arguments, output=output))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )

            if choice.finish_reason == "stop":
                break

        return steps, messages, None

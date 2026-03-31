"""Built-in general-purpose ReAct agent flow.

Uses TaskSpec (when available) to adapt its system prompt and input rendering
to any benchmark, replacing all single-purpose agent flows.

Runs a multi-turn ReAct loop: the model can emit tool calls which are
dispatched and fed back as tool-result messages.  When no tools are
available (the common case for most benchmarks) the loop naturally
completes in a single turn.
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from rllm.experimental.eval.types import AgentConfig, Task
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = "Solve the given task."
_MAX_TURNS = 16


class ReactAgentFlow:
    """General-purpose multi-turn ReAct agent driven by TaskSpec.

    On each turn the model may return tool calls.  If tools were provided
    (via ``config.metadata["tools"]``), the agent executes them and
    appends the results before the next turn.  The loop ends when:

    * the model responds with no tool calls (final answer), or
    * ``max_turns`` is reached.

    For benchmarks that set no tools the first response is the final
    answer, giving single-turn behaviour with zero overhead.
    """

    def __init__(self, max_turns: int = _MAX_TURNS):
        self.max_turns = max_turns

    def run(self, task: Task, config: AgentConfig) -> Episode:
        spec = task.spec
        system_prompt = spec.instruction if spec else _DEFAULT_SYSTEM
        user_content = spec.render_input(task.data) if spec else task.data.get("question", "")

        client = OpenAI(base_url=config.base_url, api_key="EMPTY")

        # Tools supplied by the caller (list of OpenAI tool dicts).
        # Each entry must also carry an "execute" callable under the
        # key "_execute" for the agent to dispatch.
        tools_meta: list[dict] = config.metadata.get("tools", [])
        tool_schemas = [t for t in tools_meta if "function" in t] or None
        tool_executors: dict[str, callable] = {t["function"]["name"]: t["_execute"] for t in tools_meta if "_execute" in t}

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        steps: list[Step] = []
        last_content = ""
        input_text = user_content if isinstance(user_content, str) else task.data.get("question", "")

        try:
            for turn in range(self.max_turns):
                kwargs: dict = {"model": config.model, "messages": messages}
                if tool_schemas:
                    kwargs["tools"] = tool_schemas

                response = client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                assistant_content = choice.message.content or ""
                last_content = assistant_content
                tool_calls = choice.message.tool_calls or []

                # Build assistant message for history
                assistant_msg: dict = {"role": "assistant", "content": assistant_content}
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ]
                messages.append(assistant_msg)

                # No tool calls → final answer
                if not tool_calls:
                    steps.append(Step(input=input_text, output=assistant_content, done=True))
                    break

                # Dispatch tool calls
                for tc in tool_calls:
                    fn_name = tc.function.name
                    executor = tool_executors.get(fn_name)
                    if executor is None:
                        tool_result = f"Error: unknown tool '{fn_name}'"
                    else:
                        try:
                            args = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        try:
                            tool_result = executor(**args)
                        except Exception as e:
                            tool_result = f"Error: {e}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(tool_result),
                        }
                    )

                steps.append(Step(input=input_text, output=assistant_content, done=False))
            else:
                # max_turns exhausted — mark last step done
                if steps and not steps[-1].done:
                    steps[-1] = Step(input=steps[-1].input, output=steps[-1].output, done=True)

        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            if not steps:
                steps.append(Step(input=input_text, output="", done=True))

        traj = Trajectory(name="solver", steps=steps)
        return Episode(task=task.data, trajectories=[traj], artifacts={"answer": last_content})


react_agent = ReactAgentFlow()

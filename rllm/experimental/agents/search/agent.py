"""Built-in multi-turn search agent flow with web search tool calling.

Closely follows the reference implementation in
``examples/fully_async/deepresearch/search_agent.py`` but adapted for the
eval framework (synchronous, returns Episode, no tokenizer/training deps).
"""

from __future__ import annotations

import json
import logging
import time

from openai import OpenAI

from rllm.experimental.agents.search.backends import resolve_search_backend
from rllm.experimental.eval.types import AgentConfig, Task
from rllm.rewards.math_utils.utils import extract_boxed_answer
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)

SEARCH_SYSTEM_PROMPT = """\
You are a helpful AI assistant that can search for information to answer questions accurately.

When answering questions:
1. Use the available search tools to find relevant and reliable information
2. Synthesize information from multiple sources when needed
3. Provide accurate and comprehensive answers based on your search results
4. Always put your final answer in \\boxed{} format

For example:
- If the answer is "American", write: \\boxed{American}
- If the answer is "yes", write: \\boxed{yes}
- If the answer is a year like "1985", write: \\boxed{1985}

Remember to search thoroughly and provide your final answer clearly within the \\boxed{} format."""


class SearchAgentFlow:
    """Multi-turn search agent flow implementing AgentFlow.

    Sends the user question with a web search tool, loops until the model
    produces a final answer (no more tool calls) or limits are hit.

    Mirrors the reference implementation's:
    - Duplicate search detection (set of (name, arguments) tuples)
    - Excessive parallel calls detection (>= 9 tool calls in one turn)
    - Tool error detection (result starts with "Error:")
    - Answer extraction via extract_boxed_answer
    - Aggregated metrics tracking
    """

    def __init__(self, max_turns: int = 16):
        self.max_turns = max_turns

    def run(self, task: Task, config: AgentConfig) -> Episode:
        task_data = task.data if isinstance(task, Task) else task
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        backend_name = config.metadata.get("search_backend")
        tool = resolve_search_backend(backend_name)

        messages = [
            {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": task_data.get("question", "")},
        ]

        steps: list[Step] = []
        per_turn_metrics: list[dict] = []

        # Track state for early stopping
        executed_search_calls: set[tuple[str, str]] = set()
        duplicate_search_detected = False
        excessive_parallel_calls = False
        tool_error_detected = False

        last_content = ""
        num_turns = 0

        try:
            for turn in range(self.max_turns):
                num_turns = turn + 1

                response = client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    tools=[tool.json],
                )

                choice = response.choices[0]
                assistant_content = choice.message.content or ""
                last_content = assistant_content

                # Build assistant message for conversation history
                assistant_msg: dict = {"role": "assistant", "content": assistant_content}
                tool_calls = choice.message.tool_calls or []

                if tool_calls:
                    # Serialize tool_calls for message history
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

                if not tool_calls or turn >= self.max_turns - 1:
                    # No tool calls or last turn — treat as final answer
                    steps.append(
                        Step(
                            input=messages[1].get("content", ""),
                            output=assistant_content,
                            done=True,
                        )
                    )
                    break

                # Check for excessive parallel tool calls
                if len(tool_calls) >= 9:
                    excessive_parallel_calls = True
                    steps.append(
                        Step(
                            input=messages[1].get("content", ""),
                            output=assistant_content,
                            done=True,
                        )
                    )
                    break

                # Check for duplicate search calls
                for tc in tool_calls:
                    call_key = (tc.function.name, tc.function.arguments)
                    if call_key in executed_search_calls:
                        duplicate_search_detected = True
                        break
                    executed_search_calls.add(call_key)

                if duplicate_search_detected:
                    steps.append(
                        Step(
                            input=messages[1].get("content", ""),
                            output=assistant_content,
                            done=True,
                        )
                    )
                    break

                # Execute tool calls
                turn_metrics = {
                    "tool_calls": 0,
                    "parse_tool_args_error": 0,
                    "tool_return_error": 0,
                    "tool_wait_time": 0.0,
                    "query_length": 0,
                }

                for tc in tool_calls:
                    # Parse arguments
                    parse_error = 0
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                        parse_error = 1

                    query = args.get("query", "")
                    turn_metrics["query_length"] += len(query)
                    turn_metrics["tool_calls"] += 1
                    turn_metrics["parse_tool_args_error"] += parse_error

                    # Execute tool
                    tool_start = time.time()
                    tool_result = tool.run(query)
                    tool_wait = time.time() - tool_start
                    turn_metrics["tool_wait_time"] += tool_wait

                    if tool_result.startswith("Error:"):
                        turn_metrics["tool_return_error"] += 1
                        tool_error_detected = True

                    # Append tool response message
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result,
                        }
                    )

                per_turn_metrics.append(turn_metrics)

                # Record step for this turn
                steps.append(
                    Step(
                        input=query,
                        output=assistant_content,
                        done=False,
                    )
                )

                if tool_error_detected:
                    break

        except Exception as e:
            logger.warning("Search agent error: %s", e)
            if not steps:
                steps.append(Step(input=task_data.get("question", ""), output="", done=True))

        # Mark last step as done
        if steps and not steps[-1].done:
            steps[-1] = Step(input=steps[-1].input, output=steps[-1].output, done=True)

        # Extract final answer
        final_answer = ""
        if last_content:
            boxed = extract_boxed_answer(last_content)
            if boxed:
                final_answer = boxed
            else:
                final_answer = last_content

        # Aggregate metrics
        total_tool_calls = sum(m.get("tool_calls", 0) for m in per_turn_metrics)
        aggregated_metrics = {
            "num_turns": num_turns,
            "total_tool_calls": total_tool_calls,
            "total_parse_tool_args_error": sum(m.get("parse_tool_args_error", 0) for m in per_turn_metrics),
            "total_tool_return_error": sum(m.get("tool_return_error", 0) for m in per_turn_metrics),
            "total_tool_wait_time": sum(m.get("tool_wait_time", 0.0) for m in per_turn_metrics),
            "total_query_length": sum(m.get("query_length", 0) for m in per_turn_metrics),
            "duplicate_search_detected": duplicate_search_detected,
            "excessive_parallel_calls": excessive_parallel_calls,
            "tool_error_detected": tool_error_detected,
        }

        traj = Trajectory(name="search", steps=steps)
        return Episode(
            task=task_data,
            trajectories=[traj],
            artifacts={
                "answer": final_answer,
                "metrics": aggregated_metrics,
                "conversation": [{"role": m.get("role", ""), "content": m.get("content", "")} for m in messages if m.get("role") in ("assistant", "user", "system")],
            },
        )


# Singleton instance for registry
search_agent = SearchAgentFlow()

"""FinQA AgentFlow.

Multi-turn ReAct-style agent that answers SEC-10K financial questions by
calling four tools (``get_table_names``, ``get_table_info``, ``sql_query``,
``calculator``) and concluding with a ``FINAL ANSWER:`` block.

Uses native OpenAI function calling — no qwen-style ``<tool_call>`` text
parsing. The full assistant response (including the FINAL ANSWER text)
ends up in ``episode.artifacts["answer"]``; the list of tables the model
inspected via ``get_table_info`` is recorded in
``episode.artifacts["accessed_tables"]`` so the evaluator can score the
table-access bonus.
"""

from __future__ import annotations

import json
import logging

from finqa_constants import REACT_SYSTEM_PROMPT_PATH
from finqa_tools import TOOL_FNS, TOOL_SPECS
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

logger = logging.getLogger(__name__)


with open(REACT_SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()


MAX_TURNS = 20
MAX_TOOL_OUTPUT_CHARS = 8000  # truncate over-long tool outputs to keep context bounded


def _msg_to_dict(msg) -> dict:
    """Serialize an OpenAI message object to a plain dict for replay."""
    if isinstance(msg, dict):
        return msg
    d: dict = {"role": msg.role}
    if msg.content:
        d["content"] = msg.content
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    if getattr(msg, "tool_call_id", None):
        d["tool_call_id"] = msg.tool_call_id
    return d


def _truncate(s: str, n: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(s) <= n:
        return s
    return s[: n // 2] + f"\n…(truncated {len(s) - n} chars)…\n" + s[-n // 2 :]


def _exec_tool_call(tc, accessed_tables: list[str]) -> str:
    """Dispatch a single tool call to the matching python function. Always
    returns a string (errors included) so the chat-message tool reply is well-formed."""
    name = tc.function.name
    fn = TOOL_FNS.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'. Valid tools: {list(TOOL_FNS)}"

    try:
        args = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError as e:
        return f"Error: failed to parse tool arguments: {e}"

    # Track accessed tables for the table-access reward bonus.
    if name == "get_table_info":
        t = args.get("table_name")
        if isinstance(t, str) and t.strip():
            accessed_tables.append(t.lower().strip())

    try:
        result = fn(**args)
    except TypeError as e:
        return f"Error: bad arguments for {name}: {e}"
    except Exception as e:
        return f"Error: {name} raised {type(e).__name__}: {e}"

    return _truncate(str(result))


@rllm.rollout(name="finqa")
async def finqa_flow(task: Task, config: AgentConfig) -> Episode:
    """Multi-turn tool-calling flow for financial QA."""
    meta = task.metadata or {}
    question = str(meta.get("question") or task.instruction or "")

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    accessed_tables: list[str] = []
    steps: list[Step] = []
    final_response = ""

    for turn in range(MAX_TURNS):
        try:
            resp = await client.chat.completions.create(
                model=config.model,
                messages=messages,
                tools=TOOL_SPECS,
                temperature=config.sampling_params.get("temperature", 0.6),
                max_tokens=config.sampling_params.get("max_tokens", 4096),
                timeout=300,
            )
        except Exception as e:
            logger.warning("finqa task %s turn %d: LLM call failed: %s", task.id, turn, e)
            break

        msg = resp.choices[0].message
        content = msg.content or ""
        tool_calls = msg.tool_calls or []

        messages.append(_msg_to_dict(msg))
        steps.append(
            Step(
                chat_completions=list(messages),
                model_response=content,
                action=content,
                thought=content,
            )
        )

        if not tool_calls:
            # Agent stopped calling tools — its message IS the final answer.
            final_response = content
            break

        for tc in tool_calls:
            output = _exec_tool_call(tc, accessed_tables)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
    else:
        # MAX_TURNS exhausted — use the last assistant content as the answer.
        final_response = steps[-1].model_response if steps else ""

    return Episode(
        trajectories=[Trajectory(name="finqa", steps=steps)],
        artifacts={
            "answer": final_response,
            "accessed_tables": accessed_tables,
            "turns": len(steps),
        },
    )

"""Multi-turn math agent with calculator tool.

A multi-turn agent that solves math problems step by step using a calculator
tool via native OpenAI function-calling.  Works identically for eval and
training (the gateway handles trace capture).
"""

from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

import rllm
from rllm.experimental.eval.types import AgentConfig, Task
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)

MAX_TURNS = 8

SYSTEM_PROMPT = """\
You are a math assistant that solves competition math problems step by step.
You have access to a calculator tool and you MUST use it.

IMPORTANT rules you must follow:
1. You MUST call the calculator tool at least once before giving your final answer. \
Answers given without any prior tool call will be marked wrong.
2. Do NOT perform arithmetic in your head — every computation must go through the calculator.
3. Break the problem into small steps. Make one tool call per step, then reason about the result.
4. When you have the final answer, put it in \\boxed{ANSWER} in your response.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression. Supports +, -, *, /, ** (power), % (modulo), and parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate, e.g. '(3 + 4) * 2'",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


def _safe_eval(expression: str) -> str:
    """Safely evaluate an arithmetic expression.

    Only digits, operators, parentheses, dots, and whitespace are allowed.
    """
    if len(expression) > 100:
        return "Error: expression too long"
    allowed = set("0123456789+-*/().  %")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters in expression"
    try:
        result = eval(expression)
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(round(result, 6))
    except Exception as e:
        return f"Error: {e}"


def _extract_answer(text: str) -> str:
    """Try multiple patterns to extract the final answer.

    Order: \\boxed{} → <answer> tags → #### → "answer is X" → last number.
    """

    # 1. \boxed{ANSWER}
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()

    # 2. <answer>ANSWER</answer>
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 3. #### ANSWER
    m = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()

    # 4. "the answer is NUMBER" / "the final answer is NUMBER"
    m = re.search(r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*([+-]?\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")

    # 5. Last standalone number in the text (aggressive fallback)
    numbers = re.findall(r"(?<!\w)([+-]?\d[\d,]*\.?\d*)(?!\w)", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def _msg_to_dict(msg) -> dict:
    """Convert an OpenAI message object to a plain dict for serialization."""
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


@rllm.rollout(name="math-tool-agent")
def math_tool_agent(task: Task, config: AgentConfig) -> Episode:
    """Multi-turn agent that solves math problems using a calculator tool."""
    client = OpenAI(base_url=config.base_url, api_key="EMPTY")
    question = task.data["question"]

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    steps = []
    final_answer = ""
    used_tool = False

    for turn in range(MAX_TURNS):
        logger.info("Task %s turn %d: calling LLM (%d messages)", question[:40], turn, len(messages))
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                tools=TOOLS,
                temperature=1.0,
                max_tokens=2048,
                timeout=120,
            )
        except Exception as e:
            logger.warning("Task %s turn %d: LLM call failed: %s", question[:40], turn, e)
            break

        msg = response.choices[0].message
        content = msg.content or ""
        tool_calls = msg.tool_calls or []

        # Append the assistant message (may include tool_calls)
        assistant_msg = _msg_to_dict(msg)
        messages.append(assistant_msg)
        logger.info("Task %s turn %d: got %d chars, %d tool calls", question[:40], turn, len(content), len(tool_calls))

        # Record this LLM call as a step
        steps.append(
            Step(
                chat_completions=[_msg_to_dict(m) if not isinstance(m, dict) else m for m in messages],
                model_response=content,
                action=content,
            )
        )

        # Execute any tool calls
        if tool_calls:
            used_tool = True
            for tc in tool_calls:
                args = json.loads(tc.function.arguments)
                expr = args.get("expression", "")
                result = _safe_eval(expr)
                logger.info("Task %s turn %d: tool(%s) = %s", question[:40], turn, expr, result)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            # No tool calls — model produced a text response; extract answer and stop.
            final_answer = _extract_answer(content) if used_tool else ""
            break

    # If loop ended without extracting (e.g. hit MAX_TURNS), try last content
    if not final_answer and used_tool and steps:
        last_content = steps[-1].action or ""
        final_answer = _extract_answer(str(last_content))

    return Episode(
        trajectories=[Trajectory(name="solver", steps=steps)],
        artifacts={"answer": final_answer},
    )

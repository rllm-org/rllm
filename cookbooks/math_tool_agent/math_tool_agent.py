"""Multi-turn math agent with calculator tool.

A multi-turn agent that solves arithmetic problems step by step using a
calculator tool.  Uses text-based tool calling with a plain OpenAI client —
works identically for eval and training (the gateway handles trace capture).
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

MAX_TURNS = 5

SYSTEM_PROMPT = """\
You are a math assistant that solves arithmetic problems step by step.
You have access to a calculator tool.

To use the calculator, output a tool call in this exact format:
<tool_call>{"name": "calculate", "arguments": {"expression": "EXPR"}}</tool_call>

where EXPR is an arithmetic expression using +, -, *, / and parentheses.

Rules:
1. Use the calculator for each arithmetic step — do not compute in your head.
2. Wait for the tool result before continuing.
3. When you have the final answer, write it as: <answer>NUMBER</answer>
"""


def _safe_eval(expression: str) -> str:
    """Safely evaluate an arithmetic expression.

    Only digits, operators, parentheses, dots, and whitespace are allowed.
    """
    if len(expression) > 100:
        return "Error: expression too long"
    allowed = set("0123456789+-*/().  ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters in expression"
    try:
        result = eval(expression)
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(round(result, 6))
    except Exception as e:
        return f"Error: {e}"


def _parse_tool_call(text: str) -> dict | None:
    """Extract a tool call from <tool_call>...</tool_call> tags."""
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def _extract_answer(text: str) -> str:
    """Try multiple patterns to extract the final numeric answer.

    Order: <answer> tags → #### (GSM8K) → \\boxed{} → last number in text.
    """

    # 1. <answer>NUMBER</answer>
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2. #### NUMBER  (GSM8K convention)
    m = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()

    # 3. \boxed{NUMBER}
    m = re.search(r"\\boxed\{([^}]+)\}", text)
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


@rllm.rollout(name="math-tool-agent")
def math_tool_agent(task: Task, config: AgentConfig) -> Episode:
    """Multi-turn agent that solves math problems using a calculator tool."""
    client = OpenAI(base_url=config.base_url, api_key="EMPTY")
    question = task.data["question"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    steps = []
    final_answer = ""

    for turn in range(MAX_TURNS):
        logger.info("Task %s turn %d: calling LLM (%d messages)", question[:40], turn, len(messages))
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=1.0,
                max_tokens=2048,
                timeout=120,
            )
        except Exception as e:
            logger.warning("Task %s turn %d: LLM call failed: %s", question[:40], turn, e)
            break

        content = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})
        logger.info("Task %s turn %d: got %d chars", question[:40], turn, len(content))

        # Record this LLM call as a step
        steps.append(
            Step(
                chat_completions=list(messages),
                model_response=content,
                action=content,
            )
        )

        # Check for final answer (strict: <answer> tag)
        m = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if m:
            final_answer = m.group(1).strip()
            break

        # Check for tool call
        tool_call = _parse_tool_call(content)
        if tool_call:
            expr = tool_call.get("arguments", {}).get("expression", "")
            result = _safe_eval(expr)
            logger.info("Task %s turn %d: tool(%s) = %s", question[:40], turn, expr, result)
            messages.append({"role": "user", "content": f"[Tool Result]\n{result}"})
        else:
            # Model didn't call a tool or give a tagged answer — stop
            break

    # Fallback: extract answer from last response using flexible patterns
    if not final_answer and steps:
        last_content = steps[-1].action or ""
        final_answer = _extract_answer(str(last_content))

    return Episode(
        trajectories=[Trajectory(name="solver", steps=steps)],
        artifacts={"answer": final_answer},
    )

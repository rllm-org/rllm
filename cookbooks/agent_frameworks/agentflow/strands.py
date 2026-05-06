"""Strands Agents math agent — a single-file AgentFlow."""

from __future__ import annotations

from calculator import safe_eval
from openai import AsyncOpenAI
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from system_prompt import SYSTEM_PROMPT

import rllm
from rllm.types import AgentConfig, Task


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate. Supports +, -, *,
            /, //, **, %, parentheses, and a fixed whitelist of functions
            (sqrt, log, sin, cos, factorial, comb, ...) and constants
            (pi, e, tau).

    Returns:
        The string form of the result, or an "Error: ..." message.
    """
    return safe_eval(expression)


@rllm.rollout(name="strands-math")
async def strands_math(task: Task, config: AgentConfig) -> None:
    """Strands Agent with a calculator tool.

    Returns None: the OpenAI-compatible client is pointed at config.base_url,
    so the rLLM model gateway captures every LLM call automatically. The
    framework auto-builds the Episode; the evaluator reads the answer from
    the trajectory's last assistant message.
    """
    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    model = OpenAIModel(client=client, model_id=config.model)
    agent = Agent(model=model, tools=[calculate], system_prompt=SYSTEM_PROMPT)
    await agent.invoke_async(task.instruction)
    return None

"""OpenAI Agents SDK math agent — a single-file AgentFlow."""

from __future__ import annotations

from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Task

from ._calculator import safe_eval
from ._system_prompt import SYSTEM_PROMPT


@function_tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return safe_eval(expression)


@rllm.rollout(name="openai-agents-math")
async def openai_agents_math(task: Task, config: AgentConfig) -> None:
    """OpenAI Agents SDK Agent with a calculator tool.

    Returns None: the AsyncOpenAI client is pointed at config.base_url, so
    the rLLM model gateway captures every LLM call automatically. The
    framework auto-builds the Episode; the evaluator reads the answer
    from the trajectory's last assistant message.
    """
    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    model = OpenAIChatCompletionsModel(model=config.model, openai_client=client)
    agent = Agent(
        name="solver",
        instructions=SYSTEM_PROMPT,
        tools=[calculate],
        model=model,
    )
    await Runner.run(agent, input=task.instruction)
    return None

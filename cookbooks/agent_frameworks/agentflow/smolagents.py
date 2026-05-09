"""smolagents math agent — a single-file AgentFlow."""

from __future__ import annotations

from smolagents import OpenAIServerModel, ToolCallingAgent, tool

import rllm
from rllm.types import AgentConfig, Task

from ._calculator import safe_eval
from ._system_prompt import SYSTEM_PROMPT


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: The expression to evaluate. Supports +, -, *, /, //, **, %,
            parentheses, and a fixed whitelist of functions (sqrt, log, sin,
            cos, factorial, comb, ...) and constants (pi, e, tau).
    """
    return safe_eval(expression)


@rllm.rollout(name="smolagents-math")
def smolagents_math(task: Task, config: AgentConfig) -> None:
    """smolagents ToolCallingAgent with a calculator tool.

    Returns None: OpenAIServerModel is pointed at config.base_url, so the
    rLLM model gateway captures every LLM call automatically. The framework
    auto-builds the Episode; the evaluator reads the answer from the
    trajectory's last assistant message.
    """
    model = OpenAIServerModel(
        model_id=config.model,
        api_base=config.base_url,
        api_key="EMPTY",
    )
    agent = ToolCallingAgent(tools=[calculate], model=model)
    agent.run(SYSTEM_PROMPT + "\n\n" + str(task.instruction))
    return None

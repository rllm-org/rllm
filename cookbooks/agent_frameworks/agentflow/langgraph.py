"""LangGraph math agent — a single-file AgentFlow."""

from __future__ import annotations

from calculator import safe_eval
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from system_prompt import SYSTEM_PROMPT

import rllm
from rllm.types import AgentConfig, Task


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return safe_eval(expression)


@rllm.rollout(name="langgraph-math")
async def langgraph_math(task: Task, config: AgentConfig) -> None:
    """LangGraph create_react_agent with a calculator tool.

    Returns None: ChatOpenAI is pointed at config.base_url, so the rLLM
    model gateway captures every LLM call automatically. The framework
    auto-builds the Episode from those traces; the evaluator pulls the
    answer from the trajectory's last assistant message.
    """
    llm = ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        api_key="EMPTY",
        temperature=1.0,
    )
    agent = create_react_agent(llm, tools=[calculate], prompt=SYSTEM_PROMPT)
    await agent.ainvoke({"messages": [("user", task.instruction)]})
    return None

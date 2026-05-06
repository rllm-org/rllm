"""LangGraph-authored math agent that trains via rLLM's AgentFlow protocol.

The agent is a LangGraph ``create_react_agent`` with a single calculator tool.
Because LangChain's ``ChatOpenAI`` accepts ``base_url``, we point it at the
gateway session URL from :class:`rllm.types.AgentConfig` and the gateway
captures every LLM call automatically — no callback handler required.

The flow returns ``None``; the framework auto-builds the Episode and the
evaluator pulls the answer out of the gateway-captured trajectory.
"""

from __future__ import annotations

import ast
import math

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import rllm
from rllm.types import AgentConfig, Task

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


# ---------------------------------------------------------------------------
# Calculator (lifted from cookbooks/math_tool_agent — same safe-eval whitelist)
# ---------------------------------------------------------------------------


_SAFE_NAMES: dict[str, object] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "comb": math.comb,
    "binom": math.comb,
    "perm": math.perm,
    "degrees": math.degrees,
    "radians": math.radians,
}

_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.List,
    ast.Tuple,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


def _safe_eval(expression: str) -> str:
    if len(expression) > 200:
        return "Error: expression too long"
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return f"Error: {e.msg}"
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return f"Error: disallowed syntax ({type(node).__name__})"
        if isinstance(node, ast.Name) and node.id not in _SAFE_NAMES:
            return f"Error: unknown name '{node.id}'"
    try:
        result = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, _SAFE_NAMES)
    except Exception as e:
        return f"Error: {e}"
    if isinstance(result, bool):
        return str(result)
    if isinstance(result, int):
        return str(result)
    if isinstance(result, float):
        if result == int(result):
            return str(int(result))
        return str(round(result, 6))
    return str(result)


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Supports +, -, *, /, //, **, %, parentheses, and a fixed whitelist of
    functions (sqrt, log, sin, cos, factorial, comb, ...) and constants
    (pi, e, tau).
    """
    return _safe_eval(expression)


# ---------------------------------------------------------------------------
# AgentFlow
# ---------------------------------------------------------------------------


@rllm.rollout(name="langgraph-math")
async def langgraph_math(task: Task, config: AgentConfig) -> None:
    """Multi-turn math agent built with LangGraph's create_react_agent.

    Returns None: the gateway captures every LLM call (because the LLM client
    is pointed at config.base_url), and the framework auto-builds a single
    Trajectory whose Steps come from those traces. The evaluator extracts the
    answer from the last assistant message.
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

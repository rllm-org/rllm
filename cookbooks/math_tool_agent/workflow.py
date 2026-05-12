"""
MathToolWorkflow: Workflow-based multi-turn math agent with calculator tool.

This workflow implements a pipeline-based version of the multi-turn math agent
from this cookbook, corresponding to the agent defined in `math_tool_agent.py`.
It leverages the `rllm.experimental.rollout` stack directly, with the following flow:

    RolloutEngine → TITOCompleter → MessageList state

This enables stepwise multi-turn math problem-solving with explicit calculator tool calls.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
from typing import Any

from rllm.parser.messages import MessageList
from rllm.types import Episode, Trajectory
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow

logger = logging.getLogger(__name__)


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

_CALCULATOR_DESCRIPTION = (
    "Evaluate a mathematical expression. Supports operators +, -, *, /, //, ** (power), "
    "% (modulo), and parentheses. Functions: abs, round, min, max, sum, pow, sqrt, exp, "
    "log, log2, log10, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, floor, "
    "ceil, trunc, factorial, gcd, lcm, comb (aka binom), perm, degrees, radians. "
    "Constants: pi, e, tau."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": _CALCULATOR_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The expression to evaluate.",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

_SAFE_NAMES: dict[str, Any] = {
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


def _extract_task_fields(task) -> tuple[str, str]:
    """Pull (question, expected_answer) out of a task in a shape-agnostic way.

    Dataset entries that come through ``batch.non_tensor_batch["extra_info"]``
    are dicts; their field names vary by dataset:
      * deepscaler_math/train: ``{problem, answer, solution, question, ground_truth, data_source}``
      * math500/test:           ``{question, ground_truth, data_source}``
      * legacy/SDK callers:     ``Task`` dataclass with ``.instruction``
    We prefer ``question`` → ``instruction`` → ``problem`` → ``prompt``;
    and ``answer`` → ``ground_truth``. The AgentFlow evaluator uses the same
    fallback chain (see ``cookbooks/math_tool_agent/math_tool_eval.py``).
    """
    if isinstance(task, dict):
        getter = task.get
    else:
        getter = lambda k, default=None: getattr(task, k, default)  # noqa: E731
    question = getter("question") or getter("instruction") or getter("problem") or getter("prompt") or ""
    expected_answer = getter("answer") or getter("ground_truth") or ""
    return question, str(expected_answer)


def _extract_boxed(text: str) -> str | None:
    idx = text.rfind(r"\boxed{")
    if idx < 0:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def _extract_answer(text: str) -> str:
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*([+-]?\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    numbers = re.findall(r"(?<!\w)([+-]?\d[\d,]*\.?\d*)(?!\w)", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


class MathToolWorkflow(Workflow):
    """Workflow that solves a math task via a multi-turn calculator-tool loop.

    The conversation state is a ``MessageList``. Each turn:

    1. Render the current message list, run the engine via the completer,
       and capture a Step (with prompt_ids + completion_ids for training).
    2. Parse the model's response into ``(content, reasoning, tool_calls)``.
       Append the assistant message to the MessageList.
    3. If the model emitted tool calls, execute each one and append the
       tool-response message; loop back to (1).
    4. If the model did NOT emit tool calls, this is the final turn —
       extract the boxed answer and break.

    Reward: 1.0 if the extracted answer matches ``task["answer"]``, else 0.0.
    The reward is assigned to the LAST step in the trajectory so the
    advantage estimator can credit-assign across the multi-turn rollout.
    """

    def __init__(
        self,
        rollout_engine,
        executor,
        max_turns: int = 8,
        **kwargs,
    ):
        super().__init__(rollout_engine=rollout_engine, executor=executor, **kwargs)
        self.max_turns = max_turns

    def _build_assistant_message(self, model_output) -> dict:
        msg: dict = {"role": "assistant", "content": model_output.content or ""}
        if model_output.reasoning:
            msg["reasoning"] = model_output.reasoning
        if model_output.tool_calls:
            # Build OpenAI-shape tool_calls dicts. rLLM ToolCall has (name, arguments dict);
            # we serialize the arguments back to a JSON string for the wire format.
            tcs = []
            for i, tc in enumerate(model_output.tool_calls):
                tcs.append(
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                )
            msg["tool_calls"] = tcs
        return msg

    def _compute_reward(self, final_text: str, expected_answer: str) -> tuple[float, bool]:
        """Grade the final answer. Uses the same path as the AgentFlow evaluator
        (math_tool_eval.py) so LaTeX / symbolic forms are normalized before
        comparison — plain string-equals under-rewards correct answers like
        \\frac{1}{2} vs 0.5.
        """
        predicted = _extract_answer(final_text)
        is_correct = False
        if expected_answer:
            try:
                from rllm.rewards.math_utils.utils import grade_answer_mathd, grade_answer_sympy

                is_correct = grade_answer_mathd(predicted, str(expected_answer)) or grade_answer_sympy(predicted, str(expected_answer))
            except Exception as exc:  # noqa: BLE001
                logger.warning("symbolic grading failed (%s); falling back to string-eq", exc)
                is_correct = predicted.strip() == str(expected_answer).strip()
        return (1.0 if is_correct else 0.0), is_correct

    def _finalize_trajectory(self, task, steps, final_text, expected_answer) -> None:
        """Compute the reward, attach it to the last step, and commit the
        trajectory. Idempotent in the sense that it's safe to call exactly
        once at the end of run() regardless of termination reason — every
        exit path (clean break, max-turns, length-exceeded) goes through here
        in the run()'s ``finally`` block so a partial trajectory still gets
        captured for training.
        """
        if not steps:
            return  # nothing to commit; engine never returned even once
        reward, _ = self._compute_reward(final_text, expected_answer)
        steps[-1].reward = reward
        steps[-1].done = True
        traj = Trajectory(name="solver", steps=steps, task=task)
        traj.reward = reward
        self.commit(name="solver", trajectory=traj)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        # Lazy import to avoid pulling experimental modules at workflow-import time.
        from rllm.experimental.rollout.completer import TITOCompleter

        question, expected_answer = _extract_task_fields(task)

        completer = TITOCompleter(self.rollout_engine)

        messages = MessageList()
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": question})

        steps = []
        final_text = ""
        pending_termination: TerminationEvent | None = None

        try:
            for turn in range(self.max_turns):
                step = await completer.complete(
                    messages.to_list(),
                    tools=TOOLS,
                    **kwargs,
                )
                steps.append(step)

                model_output = step.model_output
                content = (model_output.content or "") if model_output is not None else ""
                tool_calls = (model_output.tool_calls or []) if model_output is not None else []
                final_text = content

                messages.append(self._build_assistant_message(model_output))

                if not tool_calls:
                    # No more tool calls — this is the final turn.
                    break

                # Run the calculator for each tool call and append tool responses.
                for i, tc in enumerate(tool_calls):
                    # Defensive: tc.arguments is normally a dict (QwenToolParser parses
                    # the JSON), but malformed tool calls can leave it as a raw string.
                    args = tc.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    if not isinstance(args, dict):
                        args = {}
                    expr = args.get("expression", "")
                    result = _safe_eval(expr) if expr else "Error: missing 'expression' argument"
                    logger.debug("turn=%d tool(%s) = %s", turn, expr, result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"call_{i}",
                            "name": tc.name,
                            "content": result,
                        }
                    )

                if model_output is not None and model_output.finish_reason == "length":
                    pending_termination = TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)
                    break
            else:
                # Loop fell through max_turns without an assistant-only final turn.
                pending_termination = TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)
        finally:
            self._finalize_trajectory(task, steps, final_text, expected_answer)

        if pending_termination is not None:
            raise pending_termination

        return None  # let run_with_termination_handling postprocess

    def reset(self, task: dict | None = None, uid: str | None = None):
        return super().reset(task, uid)

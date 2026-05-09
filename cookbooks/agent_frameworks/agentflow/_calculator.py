"""Safe arithmetic evaluator shared by every flow's calculator tool.

Whitelisted AST nodes + a fixed function/constant namespace make this
suitable for an untrusted LLM to call. Anything outside the whitelist
returns an error string the model sees and can recover from.
"""

from __future__ import annotations

import ast
import math

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

DESCRIPTION = (
    "Evaluate a mathematical expression. Supports +, -, *, /, //, **, %, "
    "parentheses, and a fixed whitelist of functions (sqrt, log, sin, cos, "
    "factorial, comb, gcd, lcm, ...) and constants (pi, e, tau)."
)


def safe_eval(expression: str) -> str:
    """Evaluate an arithmetic expression and return its string form.

    Returns an ``"Error: ..."`` string for syntactically valid but
    disallowed inputs (e.g. attribute access, unknown name) — these get
    fed back to the model so it can correct itself rather than crashing
    the rollout.
    """
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

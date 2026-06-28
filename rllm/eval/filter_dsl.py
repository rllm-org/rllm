"""A tiny, safe boolean DSL for filtering eval tasks by aggregate metrics.

Used by :mod:`rllm.eval.curation` to decide which tasks survive into an SFT
dataset. A filter is a boolean expression evaluated *per task* over that task's
pooled attempts, e.g.::

    "solved"                      # >= 1 successful attempt
    "0 < avg < 1"                 # difficulty band (avg is k-invariant)
    "pass@4 >= 0.5"               # solvable >=50% of the time within 4 tries
    "best == 1 and avg < 0.5"     # solvable but usually fails

Mechanics
---------
``<name>@<k>`` tokens (``pass@4``) are rewritten to a single whitelisted accessor
call (``_at("pass", 4)``) before parsing, so the metric stays expressible with a
budget while remaining valid Python. The resulting AST is then validated against
a node whitelist — comparisons, boolean/unary ops, numeric/bool literals, the
whitelisted names, and the single ``_at`` call. No attribute access, no other
calls, no names outside the whitelist. Evaluation runs with an empty
``__builtins__`` against a namespace the caller supplies per task.

The available names are documented on :data:`ALLOWED_NAMES`; the caller
(:mod:`rllm.eval.curation`) is responsible for binding them to real values.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

# Names a filter expression may reference, bound per-task by the caller.
#   avg        mean of the chosen metric over the task's attempts (k-invariant)
#   best/worst observed max/min of the metric
#   solved     bool: at least one successful attempt
#   n          number of attempts
#   n_correct  number of successful attempts
#   _at        accessor for "<name>@<k>" forms (e.g. pass@k); injected by the rewrite
ALLOWED_NAMES = frozenset({"avg", "best", "worst", "solved", "n", "n_correct", "_at"})

# "<name>@<k>"  ->  _at("<name>", <k>)
_AT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)@(\d+)\b")

# AST node types permitted in a filter expression.
_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Compare,
    ast.Load,
    ast.Name,
    ast.Constant,
    ast.Call,
    # comparison operators
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


class FilterError(ValueError):
    """Raised when a filter expression is malformed or references unknown names."""


@dataclass
class CompiledFilter:
    """A validated, reusable filter expression.

    Evaluate it against a per-task ``namespace`` (a dict binding the names in
    :data:`ALLOWED_NAMES`). Returns ``bool``.
    """

    source: str
    _code: object

    def evaluate(self, namespace: dict) -> bool:
        try:
            result = eval(self._code, {"__builtins__": {}}, namespace)  # noqa: S307 - AST whitelisted at compile time
        except Exception as e:  # pragma: no cover - defensive; arithmetic/type errors
            raise FilterError(f"Error evaluating filter {self.source!r}: {e}") from e
        return bool(result)


def _rewrite_at_tokens(expr: str) -> str:
    """Rewrite ``name@k`` budget tokens to ``_at("name", k)`` calls."""
    return _AT_RE.sub(lambda m: f'_at("{m.group(1)}", {m.group(2)})', expr)


def _validate(tree: ast.AST, source: str) -> None:
    # The "<name>@<k>" rewrite produces _at("<name>", <k>) calls. Approve those
    # injected string name-args so the generic "numbers/bools only" rule below
    # can still reject user-written string literals.
    approved_strings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id == "_at"):
                raise FilterError(f"Filter {source!r}: function calls are not allowed.")
            if node.keywords or len(node.args) != 2:
                raise FilterError(f"Filter {source!r}: malformed metric@k expression.")
            name_arg = node.args[0]
            if not (isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str)):
                raise FilterError(f"Filter {source!r}: malformed metric@k expression.")
            approved_strings.add(id(name_arg))

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise FilterError(f"Filter {source!r}: disallowed syntax {type(node).__name__!r}.")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
            allowed = ", ".join(sorted(n for n in ALLOWED_NAMES if n != "_at"))
            raise FilterError(f"Filter {source!r}: unknown name {node.id!r}. Available: {allowed}, plus name@k forms (e.g. pass@8).")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or isinstance(node.value, int | float):
                continue
            if isinstance(node.value, str) and id(node) in approved_strings:
                continue
            raise FilterError(f"Filter {source!r}: only numeric/bool literals are allowed.")


def compile_filter(expr: str) -> CompiledFilter:
    """Compile a filter expression into a reusable :class:`CompiledFilter`.

    Raises :class:`FilterError` if the expression is empty, unparseable, uses
    disallowed syntax, or references unknown names.
    """
    if expr is None or not str(expr).strip():
        raise FilterError("Empty filter expression.")
    source = str(expr).strip()
    rewritten = _rewrite_at_tokens(source)
    try:
        tree = ast.parse(rewritten, mode="eval")
    except SyntaxError as e:
        raise FilterError(f"Could not parse filter {source!r}: {e}") from e
    _validate(tree, source)
    code = compile(tree, "<rllm-filter>", "eval")
    return CompiledFilter(source=source, _code=code)

"""Extract Miles' flag surface from source, without importing miles.

Importing ``miles.utils.arguments`` pulls in sglang (and, on the megatron path,
Megatron-LM), so it only works inside the Miles container image. Parsing the
source with ``ast`` lets the config bridge be audited from any checkout, which is
what ``tests/test_miles_flags.py`` uses to catch flag-name drift.

This is an auditing aid, not a parser: ``miles_arity()`` in ``miles_config`` is
still the authority at runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path

from rllm.trainer.miles.miles_config import BOOL_OPTIONAL

# Miles registers flags two ways: plain add_argument, and reset_arg(parser, "--flag", ...)
# which overrides a Megatron default and falls back to add_argument when the flag
# does not exist yet (as on the FSDP path, where Megatron never ran).
_REGISTRARS = {"add_argument", "reset_arg"}

# Files holding Miles' own (statically declared) flags. The --sglang-* family is
# generated at runtime from ServerArgs.add_cli_args and is deliberately not covered.
SOURCE_RELPATHS = (
    "miles/utils/arguments.py",
    "miles/backends/fsdp_utils/arguments.py",
    "miles/backends/sglang_utils/arguments.py",
    "miles/dashboard/args.py",
)


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _fsdp_dataclass_flags(root: Path) -> dict[str, int | str]:
    """Flags Miles' FSDP parser derives from the ``FSDPArgs`` dataclass.

    ``build_fsdp_parser`` walks ``dataclasses.fields(FSDPArgs)`` and registers one
    flag per field -- bools via ``BooleanOptionalAction`` -- so none of them appear
    as an ``add_argument`` call and a source scan misses them entirely.
    """
    path = root / "miles" / "backends" / "fsdp_utils" / "arguments.py"
    if not path.exists():
        return {}

    out: dict[str, int | str] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.ClassDef) and node.name == "FSDPArgs"):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            name = stmt.target.id
            if name == "config":
                continue  # build_fsdp_parser skips it
            is_bool = isinstance(stmt.annotation, ast.Name) and stmt.annotation.id == "bool"
            out[name] = BOOL_OPTIONAL if is_bool else 1
    return out


def flag_arity_from_source(miles_root: str | Path) -> dict[str, int | str]:
    """Map flag name (snake_case, no leading --) to value count: 0, 1, or "+"."""
    root = Path(miles_root)
    arity: dict[str, int | str] = _fsdp_dataclass_flags(root)

    for relpath in SOURCE_RELPATHS:
        path = root / relpath
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and _call_name(node) in _REGISTRARS):
                continue
            names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--")]
            if not names:
                continue
            kwargs = {k.arg: k.value for k in node.keywords}

            n: int | str = 1
            action = kwargs.get("action")
            if isinstance(action, ast.Constant) and action.value in ("store_true", "store_false"):
                n = 0
            nargs = kwargs.get("nargs")
            if isinstance(nargs, ast.Constant) and nargs.value in ("+", "*"):
                n = "+"

            for name in names:
                arity[name[2:].replace("-", "_")] = n

    return arity

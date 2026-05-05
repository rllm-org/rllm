"""``~/.rllm/console.env`` read/write — the persistence layer for the
Settings panel's env-var management.

Format is a tiny subset of the ``.env`` convention: ``KEY=VALUE`` per
line, ``#`` comment lines preserved, blank lines preserved. Values may
be quoted (single or double); writes always emit unquoted unless the
value contains characters that need escaping (in which case we
double-quote and JSON-escape).

Order is preserved across read → write so manual edits to the file
don't get reshuffled. Updating an existing key edits in place; new
keys append at the end.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_KEY_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def default_env_path() -> Path:
    home = Path(os.path.expanduser(os.environ.get("RLLM_HOME", "~/.rllm")))
    return home / "console.env"


def _strip_quotes(s: str) -> str:
    """Inverse of :func:`_quote_for_write`.

    Double-quoted values are JSON-decoded so escapes round-trip
    (``\\"``, ``\\n``, etc). Single-quoted values are taken literally —
    matches POSIX shell convention. Bare values pass through unchanged.
    """
    s = s.strip()
    if len(s) >= 2:
        if s[0] == s[-1] == '"':
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return s[1:-1]
        if s[0] == s[-1] == "'":
            return s[1:-1]
    return s


def _quote_for_write(value: str) -> str:
    """Emit a shell-safe representation. Plain text stays bare; values
    with spaces, ``#``, ``=``, or quotes get JSON-quoted."""
    if value == "" or any(c.isspace() or c in "#\"'\\" for c in value):
        return json.dumps(value)
    return value


def read_file(path: Path | None = None) -> dict[str, str]:
    """Return ``{key: value}`` for every assignment in the file, or
    ``{}`` if it doesn't exist or is unreadable."""
    p = path or default_env_path()
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _KEY_LINE.match(line)
        if not m:
            continue
        out[m.group(1)] = _strip_quotes(m.group(2))
    return out


def write_assignment(key: str, value: str, *, path: Path | None = None) -> None:
    """Set ``key=value`` in the env file, preserving existing lines.

    Creates parent dirs and the file as needed. Replaces the line for
    an existing key (preserving file order); appends otherwise.
    """
    p = path or default_env_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    new_line = f"{key}={_quote_for_write(value)}"

    lines: list[str]
    if p.is_file():
        lines = p.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    replaced = False
    for i, raw in enumerate(lines):
        m = _KEY_LINE.match(raw.strip())
        if m and m.group(1) == key:
            lines[i] = new_line
            replaced = True
            break
    if not replaced:
        lines.append(new_line)

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def delete_assignment(key: str, *, path: Path | None = None) -> bool:
    """Remove ``key`` from the file. Returns True if a line was removed."""
    p = path or default_env_path()
    if not p.is_file():
        return False
    lines = p.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    removed = False
    for raw in lines:
        m = _KEY_LINE.match(raw.strip())
        if m and m.group(1) == key:
            removed = True
            continue
        out.append(raw)
    if removed:
        p.write_text(("\n".join(out) + "\n") if out else "", encoding="utf-8")
    return removed


def load_into_environ(path: Path | None = None, *, override: bool = True) -> dict[str, str]:
    """Apply file values to ``os.environ``. Returns the loaded mapping.

    With ``override=True`` (default), file values win over shell-set
    values — matches the user expectation that "I set it in the UI →
    it takes effect". Pass ``override=False`` to behave like
    ``python-dotenv``'s default.
    """
    pairs = read_file(path)
    for k, v in pairs.items():
        if override or k not in os.environ:
            os.environ[k] = v
    return pairs

"""Typed readers for ``RLLM_*`` operational environment variables.

These knobs tune *infrastructure* behaviour — timeouts, retries, TTLs — that an
operator may need to change per environment (CI vs. cluster vs. laptop) without
editing code or touching the experiment config. They are deliberately env vars
rather than config fields: config is for experiment design, env vars are for the
operational surface. See ``docs/guides/environment-variables.mdx`` for the catalogue.

The environment is read when the helper is called. Call it at module import to
get a constant (the common case), or per-use if you need a live value. Every
``RLLM_*`` name is forwarded to verl training workers automatically via
``FORWARD_PREFIXES`` in ``rllm/trainer/verl/ray_runtime_env.py``, so a worker that
re-imports the module sees the value the operator set on the launching node.

Prefer these helpers over re-deriving ``os.getenv("RLLM_X", ...)`` inline so the
parsing (and the ``RLLM_`` naming convention) lives in one place.
"""

from __future__ import annotations

import os
import re

_RUN_ID: str | None = None


def rllm_run_id() -> str:
    """Stable per-process run identifier stamped onto cloud sandboxes.

    Used as the Modal App name suffix and a sandbox tag so a single run's
    sandboxes are greppable and terminable on a *shared* account (where you
    can't just kill every ``rllm-*`` sandbox because others' runs use them).

    Set ``RLLM_RUN_ID`` to a memorable value (e.g. ``alice-tb2-0621``);
    otherwise a random 8-char id is generated once per process. Sanitized to
    Modal's label charset (``[a-zA-Z0-9_.-]``).
    """
    global _RUN_ID
    if _RUN_ID is None:
        raw = re.sub(r"[^a-zA-Z0-9_.-]", "-", os.environ.get("RLLM_RUN_ID", "")).strip("-")
        _RUN_ID = raw or os.urandom(4).hex()
    return _RUN_ID


def env_str(name: str, default: str) -> str:
    """Return ``$name`` if set and non-empty, else ``default``."""
    value = os.environ.get(name)
    return value if value else default


def env_int(name: str, default: int) -> int:
    """Return ``$name`` parsed as ``int`` if set and non-empty, else ``default``.

    Raises ``ValueError`` on a malformed value — a typo in an ops knob should fail
    loudly rather than silently fall back to the default.
    """
    value = os.environ.get(name)
    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    """Return ``$name`` parsed as ``float`` if set and non-empty, else ``default``.

    Raises ``ValueError`` on a malformed value (see :func:`env_int`).
    """
    value = os.environ.get(name)
    return float(value) if value else default

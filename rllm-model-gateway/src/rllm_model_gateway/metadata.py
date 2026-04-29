"""rLLM session/harness metadata extraction.

Defines the canonical contract for stamping session identity on a model
call: HTTP headers (preferred), request-body fallback, and the legacy
``/sessions/{sid}/v1/...`` URL path (for training back-compat).

Precedence: header > body > URL path. The first non-empty value wins
per field.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

# ------------------------------------------------------------------
# Header names (lowercase — HTTP headers are case-insensitive but
# ASGI delivers them in lowercase)
# ------------------------------------------------------------------

HEADER_SESSION_ID = "x-rllm-session-id"
HEADER_RUN_ID = "x-rllm-run-id"
HEADER_HARNESS = "x-rllm-harness"
HEADER_STEP_ID = "x-rllm-step-id"
HEADER_PARENT_SPAN_ID = "x-rllm-parent-span-id"
HEADER_PROJECT = "x-rllm-project"
HEADER_EXPERIMENT = "x-rllm-experiment"


_SESSION_PATH_RE = re.compile(r"/sessions/([^/]+)(?:/v1(?:/.*)?)?$")


class RllmMetadata(BaseModel):
    """Session-level identity stamped on a model call."""

    session_id: str | None = None
    run_id: str | None = None
    harness: str | None = None
    step_id: int | None = None
    parent_span_id: str | None = None
    project: str | None = None
    experiment: str | None = None


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def extract_metadata(
    *,
    headers: Mapping[str, str] | None = None,
    body: dict[str, Any] | None = None,
    path: str | None = None,
) -> RllmMetadata:
    """Build ``RllmMetadata`` from headers, body, and URL path.

    Args:
        headers: Lowercased header name → value. Empty values are ignored.
        body: Parsed JSON body. Looks for ``metadata.rllm`` (a dict).
            Also accepts a top-level ``rllm`` key for clients that don't
            nest under ``metadata``.
        path: Original request path. Used only as a last-resort fallback
            for ``session_id`` via the legacy ``/sessions/{sid}/...`` shape.

    Precedence is field-by-field: header → body → URL path. Missing
    values stay ``None`` rather than raising.
    """
    headers = headers or {}
    body = body or {}

    body_rllm: dict[str, Any] = {}
    md = body.get("metadata")
    if isinstance(md, dict) and isinstance(md.get("rllm"), dict):
        body_rllm = md["rllm"]
    elif isinstance(body.get("rllm"), dict):
        body_rllm = body["rllm"]

    def _str_field(header_name: str, body_key: str) -> str | None:
        return _coerce_str(headers.get(header_name)) or _coerce_str(body_rllm.get(body_key))

    session_id = _str_field(HEADER_SESSION_ID, "session_id")
    run_id = _str_field(HEADER_RUN_ID, "run_id")
    harness = _str_field(HEADER_HARNESS, "harness")
    parent_span_id = _str_field(HEADER_PARENT_SPAN_ID, "parent_span_id")
    project = _str_field(HEADER_PROJECT, "project")
    experiment = _str_field(HEADER_EXPERIMENT, "experiment")

    step_id = _coerce_int(headers.get(HEADER_STEP_ID))
    if step_id is None:
        step_id = _coerce_int(body_rllm.get("step_id"))

    if not session_id and path:
        m = _SESSION_PATH_RE.search(path)
        if m:
            session_id = m.group(1)

    return RllmMetadata(
        session_id=session_id,
        run_id=run_id,
        harness=harness,
        step_id=step_id,
        parent_span_id=parent_span_id,
        project=project,
        experiment=experiment,
    )


def headers_from_scope(scope: dict[str, Any]) -> dict[str, str]:
    """Convert ASGI ``scope['headers']`` to a lowercase-keyed dict.

    Last value wins on duplicates — fine for the X-RLLM-* headers
    which are single-valued.
    """
    raw = scope.get("headers") or []
    out: dict[str, str] = {}
    for item in raw:
        try:
            key, value = item
        except (TypeError, ValueError):
            continue
        if isinstance(key, bytes):
            key = key.decode("latin-1")
        if isinstance(value, bytes):
            value = value.decode("latin-1")
        out[key.lower()] = value
    return out

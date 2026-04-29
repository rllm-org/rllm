"""Read-side helpers for ``<run_dir>/traces.db``.

The rLLM gateway writes ``TraceRecord`` rows to a SQLite file shared
with eval (path picked by ``EvalGatewayManager.db_path``). This module
exposes a small stdlib-only ``sqlite3`` reader so the local viewer can
load them without pulling in ``aiosqlite`` or talking to a live gateway.

Schema (mirrors ``rllm_model_gateway.store.sqlite_store``)::

    traces(trace_id TEXT PK, data TEXT, created_at REAL)
    trace_sessions(trace_id TEXT, session_id TEXT, created_at REAL,
                   PRIMARY KEY (trace_id, session_id))

``data`` is a JSON-encoded ``TraceRecord``. Both tables share
``created_at`` so the reader can rank by either; we use the junction
table's ``created_at`` since that's what the gateway indexes for
session-scoped reads.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def _connect(db_path: str | Path):
    """Open a read-only connection to *db_path*.

    Uses ``mode=ro`` URI to make sure a stale lock or write attempt from
    a misbehaving caller can't corrupt the gateway's writer. Yields
    ``None`` if the file doesn't exist yet (a viewer reading a run that
    didn't use the gateway, or a run that just started).
    """
    p = Path(db_path)
    if not p.is_file():
        yield None
        return
    uri = f"file:{p.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def list_session_ids(db_path: str | Path) -> list[str]:
    """Return every session_id that has at least one trace, ordered by first-seen ASC."""
    with _connect(db_path) as conn:
        if conn is None:
            return []
        rows = conn.execute(
            """
            SELECT session_id
            FROM trace_sessions
            GROUP BY session_id
            ORDER BY MIN(created_at) ASC
            """
        ).fetchall()
        return [r[0] for r in rows]


def session_summaries(db_path: str | Path) -> dict[str, dict[str, Any]]:
    """Return ``{session_id: {trace_count, first_at, last_at}}`` for every session."""
    with _connect(db_path) as conn:
        if conn is None:
            return {}
        rows = conn.execute(
            """
            SELECT session_id,
                   COUNT(*)        AS trace_count,
                   MIN(created_at) AS first_at,
                   MAX(created_at) AS last_at
            FROM trace_sessions
            GROUP BY session_id
            """
        ).fetchall()
        return {
            r["session_id"]: {
                "trace_count": r["trace_count"],
                "first_at": r["first_at"],
                "last_at": r["last_at"],
            }
            for r in rows
        }


def get_traces(
    db_path: str | Path,
    session_id: str,
    *,
    since: float | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return ``TraceRecord`` rows for *session_id* ordered by created_at ASC.

    Each row includes a synthetic ``_created_at`` field (the junction
    table's timestamp) so callers can advance a polling cursor without
    having to inspect ``timestamp`` inside ``data`` (which is the LLM
    request time, not the persist time, and skews on long requests).
    """
    with _connect(db_path) as conn:
        if conn is None:
            return []
        sql = [
            "SELECT t.data AS data, ts.created_at AS created_at",
            "FROM traces t",
            "INNER JOIN trace_sessions ts ON t.trace_id = ts.trace_id",
            "WHERE ts.session_id = ?",
        ]
        params: list[Any] = [session_id]
        if since is not None:
            sql.append("AND ts.created_at > ?")
            params.append(since)
        sql.append("ORDER BY ts.created_at ASC")
        if limit is not None:
            sql.append("LIMIT ?")
            params.append(limit)
        rows = conn.execute(" ".join(sql), params).fetchall()

    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            data = json.loads(r["data"])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data["_created_at"] = r["created_at"]
            out.append(data)
    return out


def count_traces(db_path: str | Path, session_id: str) -> int:
    with _connect(db_path) as conn:
        if conn is None:
            return 0
        row = conn.execute(
            "SELECT COUNT(*) FROM trace_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0] or 0)

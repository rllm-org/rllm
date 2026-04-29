"""Read-side helpers for the shared gateway sqlite store.

The rLLM gateway writes ``TraceRecord`` rows to a single sqlite file at
``~/.rllm/gateway/traces.db`` (override via ``RLLM_GATEWAY_DB``). Each
row is tagged with a ``run_id`` so multiple gateway invocations can
share the file without colliding session ids. This module exposes a
small stdlib-only ``sqlite3`` reader so the local viewer can load
traces without pulling in ``aiosqlite`` or talking to a live gateway.

Schema (mirrors ``rllm_model_gateway.store.sqlite_store``)::

    traces(trace_id TEXT PK, data TEXT, created_at REAL)
    trace_sessions(
        trace_id TEXT, session_id TEXT, run_id TEXT, created_at REAL,
        PRIMARY KEY (trace_id, session_id)
    )
    runs(run_id TEXT PK, started_at REAL, ended_at REAL, metadata TEXT)

``data`` is a JSON-encoded ``TraceRecord``. The junction table's
``created_at`` is what we order by — that's the persist time, distinct
from ``data.timestamp`` which is the LLM request time.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_DEFAULT_GATEWAY_DB = "~/.rllm/gateway/traces.db"


def default_db_path() -> Path:
    """Resolve the shared gateway db path: ``RLLM_GATEWAY_DB`` env or default."""
    return Path(os.path.expanduser(os.environ.get("RLLM_GATEWAY_DB", _DEFAULT_GATEWAY_DB)))


@contextmanager
def _connect(db_path: str | Path):
    """Open a read-only connection to *db_path*.

    Uses ``mode=ro`` URI so a stale lock or write attempt from a
    misbehaving caller can't corrupt the gateway's writer. Yields
    ``None`` if the file doesn't exist yet (gateway hasn't run, or the
    user is browsing on a fresh machine).
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


def _has_run_id_column(conn: sqlite3.Connection) -> bool:
    """Return True when ``trace_sessions`` has a ``run_id`` column.

    Lets the loader stay backwards-compatible with sqlite files written
    by an older gateway: those silently get ``run_id=""`` everywhere.
    """
    cur = conn.execute("PRAGMA table_info(trace_sessions)")
    return any(row[1] == "run_id" for row in cur.fetchall())


def _scope_clause(conn: sqlite3.Connection, run_id: str | None) -> tuple[str, list[Any]]:
    """Build ``AND ts.run_id = ?`` (or empty) honouring legacy schemas."""
    if run_id is None or not _has_run_id_column(conn):
        return "", []
    return " AND ts.run_id = ?", [run_id]


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def list_session_ids(
    db_path: str | Path,
    *,
    run_id: str | None = None,
) -> list[str]:
    """Return every session_id (in this run, or globally) ordered by first-seen ASC."""
    with _connect(db_path) as conn:
        if conn is None:
            return []
        scope, params = _scope_clause(conn, run_id)
        sql = f"""
            SELECT session_id
            FROM trace_sessions ts
            WHERE 1=1 {scope}
            GROUP BY session_id
            ORDER BY MIN(created_at) ASC
        """
        rows = conn.execute(sql, params).fetchall()
        return [r[0] for r in rows]


def session_summaries(
    db_path: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return ``{session_id: {trace_count, first_at, last_at}}`` for every session.

    Scoped to ``run_id`` when given. With ``run_id=None`` the summary is
    cross-run — same session_id from two different runs is collapsed
    into one row (use :func:`session_summaries_by_run` for the
    disambiguated form).
    """
    with _connect(db_path) as conn:
        if conn is None:
            return {}
        scope, params = _scope_clause(conn, run_id)
        sql = f"""
            SELECT session_id,
                   COUNT(*)        AS trace_count,
                   MIN(created_at) AS first_at,
                   MAX(created_at) AS last_at
            FROM trace_sessions ts
            WHERE 1=1 {scope}
            GROUP BY session_id
        """
        rows = conn.execute(sql, params).fetchall()
        return {
            r["session_id"]: {
                "trace_count": r["trace_count"],
                "first_at": r["first_at"],
                "last_at": r["last_at"],
            }
            for r in rows
        }


def session_summaries_by_run(db_path: str | Path) -> list[dict[str, Any]]:
    """Cross-run session list — every (run_id, session_id) pair with stats.

    Powers the cross-run gateway view's left pane.
    """
    with _connect(db_path) as conn:
        if conn is None:
            return []
        if not _has_run_id_column(conn):
            # Legacy schema: every row is in the unstamped bucket.
            sql = """
                SELECT session_id,
                       '' AS run_id,
                       COUNT(*) AS trace_count,
                       MIN(created_at) AS first_at,
                       MAX(created_at) AS last_at
                FROM trace_sessions
                GROUP BY session_id
                ORDER BY MIN(created_at) DESC
            """
        else:
            sql = """
                SELECT session_id,
                       run_id,
                       COUNT(*) AS trace_count,
                       MIN(created_at) AS first_at,
                       MAX(created_at) AS last_at
                FROM trace_sessions
                GROUP BY session_id, run_id
                ORDER BY MIN(created_at) DESC
            """
        rows = conn.execute(sql).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "run_id": r["run_id"],
                "trace_count": r["trace_count"],
                "first_at": r["first_at"],
                "last_at": r["last_at"],
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


def get_traces(
    db_path: str | Path,
    session_id: str,
    *,
    since: float | None = None,
    limit: int | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return ``TraceRecord`` rows for *session_id* ordered by created_at ASC.

    Each row includes a synthetic ``_created_at`` (the junction table's
    timestamp) so callers can advance a polling cursor without inspecting
    ``timestamp`` inside ``data`` (which is the LLM request time and
    skews on long requests).
    """
    with _connect(db_path) as conn:
        if conn is None:
            return []
        scope, scope_params = _scope_clause(conn, run_id)
        sql = [
            "SELECT t.data AS data, ts.created_at AS created_at",
            "FROM traces t",
            "INNER JOIN trace_sessions ts ON t.trace_id = ts.trace_id",
            "WHERE ts.session_id = ?",
        ]
        params: list[Any] = [session_id, *scope_params]
        if scope:
            sql.append(scope)
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


def count_traces(
    db_path: str | Path,
    session_id: str,
    *,
    run_id: str | None = None,
) -> int:
    with _connect(db_path) as conn:
        if conn is None:
            return 0
        scope, scope_params = _scope_clause(conn, run_id)
        sql = f"SELECT COUNT(*) FROM trace_sessions ts WHERE session_id = ? {scope}"
        row = conn.execute(sql, [session_id, *scope_params]).fetchone()
        return int(row[0] or 0)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def list_runs(
    db_path: str | Path,
    *,
    since: float | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """List gateway runs ordered by ``started_at`` DESC.

    Returns ``[]`` if the db doesn't have a ``runs`` table (legacy
    schema). The cross-run viewer falls back to deriving runs from
    ``trace_sessions.run_id`` in that case.
    """
    with _connect(db_path) as conn:
        if conn is None:
            return []
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
        if cur.fetchone() is None:
            return []
        sql = "SELECT run_id, started_at, ended_at, metadata FROM runs"
        params: list[Any] = []
        if since is not None:
            sql += " WHERE started_at >= ?"
            params.append(since)
        sql += " ORDER BY started_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
        except json.JSONDecodeError:
            meta = {}
        out.append(
            {
                "run_id": r["run_id"],
                "started_at": r["started_at"],
                "ended_at": r["ended_at"],
                "metadata": meta,
            }
        )
    return out

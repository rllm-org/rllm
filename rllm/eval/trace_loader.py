"""Read-side helpers for the shared gateway sqlite store.

The rLLM gateway writes ``TraceRecord`` rows to a single sqlite file at
``~/.rllm/gateway/traces.db`` (override via ``RLLM_GATEWAY_DB``). Each
row carries denormalized filter columns (``run_id``, ``session_id``,
``model``, ``harness``, ``latency_ms``, ``has_error``, ``step_id``)
plus the JSON payload. This module is a stdlib-only ``sqlite3`` reader
so the local viewer can load traces without pulling in ``aiosqlite``
or talking to a live gateway.

Schema (mirrors ``rllm_model_gateway.store.sqlite_store`` v2)::

    traces(
        trace_id TEXT PK, data TEXT, created_at REAL,
        session_id TEXT, run_id TEXT, model TEXT, harness TEXT,
        latency_ms INTEGER, has_error INTEGER, step_id INTEGER
    )
    trace_sessions(
        trace_id TEXT, session_id TEXT, run_id TEXT, created_at REAL,
        PRIMARY KEY (trace_id, session_id)
    )
    runs(run_id TEXT PK, started_at REAL, ended_at REAL, metadata TEXT)

``data`` is a JSON-encoded ``TraceRecord``. The traces table's
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

# Must match ``rllm_model_gateway.store.sqlite_store._SCHEMA_VERSION``.
# A db at any other version is treated as missing — the gateway will
# rebuild it on next boot.
_EXPECTED_SCHEMA_VERSION = 2


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
        # Treat any non-v2 schema as "no data yet". The gateway rebuilds
        # the file on its next boot; the reader stays read-only and
        # never tries to migrate.
        version = conn.execute("PRAGMA user_version").fetchone()
        if not version or int(version[0]) != _EXPECTED_SCHEMA_VERSION:
            yield None
            return
        yield conn
    finally:
        conn.close()


def _build_filters(
    *,
    run_id: str | None,
    session_id: str | None,
    model: str | None,
    harness: str | None,
    has_error: bool | None,
    latency_min: float | None,
    latency_max: float | None,
    since: float | None,
    until: float | None,
) -> tuple[list[str], list[Any]]:
    where: list[str] = []
    params: list[Any] = []
    if run_id is not None:
        where.append("run_id = ?")
        params.append(run_id)
    if session_id is not None:
        where.append("session_id = ?")
        params.append(session_id)
    if model is not None:
        where.append("model = ?")
        params.append(model)
    if harness is not None:
        where.append("harness = ?")
        params.append(harness)
    if has_error is not None:
        where.append("has_error = ?")
        params.append(1 if has_error else 0)
    if latency_min is not None:
        where.append("latency_ms >= ?")
        params.append(latency_min)
    if latency_max is not None:
        where.append("latency_ms <= ?")
        params.append(latency_max)
    if since is not None:
        where.append("created_at > ?")
        params.append(since)
    if until is not None:
        where.append("created_at < ?")
        params.append(until)
    return where, params


# ---------------------------------------------------------------------------
# Trace queries
# ---------------------------------------------------------------------------


def query_traces(
    db_path: str | Path,
    *,
    run_id: str | None = None,
    session_id: str | None = None,
    model: str | None = None,
    harness: str | None = None,
    has_error: bool | None = None,
    latency_min: float | None = None,
    latency_max: float | None = None,
    since: float | None = None,
    until: float | None = None,
    limit: int | None = 200,
    order: str = "DESC",
) -> list[dict[str, Any]]:
    """Filter+paginate traces by their denormalized columns.

    ``since`` (newer-than) and ``until`` (older-than) are cursors on
    persisted-at time. ``order=DESC`` is the global feed default
    (newest first); ``ASC`` is the per-session timeline.

    Each returned row carries a synthetic ``_created_at`` so callers
    can advance a polling cursor without inspecting ``timestamp``
    inside ``data`` (which is LLM request time and skews on long calls).
    """
    if order not in ("ASC", "DESC"):
        raise ValueError(f"order must be ASC or DESC, got {order!r}")
    with _connect(db_path) as conn:
        if conn is None:
            return []
        where, params = _build_filters(
            run_id=run_id,
            session_id=session_id,
            model=model,
            harness=harness,
            has_error=has_error,
            latency_min=latency_min,
            latency_max=latency_max,
            since=since,
            until=until,
        )
        sql = "SELECT data, created_at FROM traces"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY created_at {order}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(sql, params).fetchall()

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
    session_id: str | None = None,
    *,
    run_id: str | None = None,
    model: str | None = None,
    harness: str | None = None,
    has_error: bool | None = None,
    latency_min: float | None = None,
    latency_max: float | None = None,
    since: float | None = None,
    until: float | None = None,
) -> int:
    """COUNT(*) over the same filters as :func:`query_traces`."""
    with _connect(db_path) as conn:
        if conn is None:
            return 0
        where, params = _build_filters(
            run_id=run_id,
            session_id=session_id,
            model=model,
            harness=harness,
            has_error=has_error,
            latency_min=latency_min,
            latency_max=latency_max,
            since=since,
            until=until,
        )
        sql = "SELECT COUNT(*) FROM traces"
        if where:
            sql += " WHERE " + " AND ".join(where)
        row = conn.execute(sql, params).fetchone()
        return int(row[0] or 0) if row else 0


def list_facets(db_path: str | Path) -> dict[str, list[str]]:
    """Distinct values for filter-bar dropdowns: ``models``, ``harnesses``, ``runs``."""
    out: dict[str, list[str]] = {"models": [], "harnesses": [], "runs": []}
    with _connect(db_path) as conn:
        if conn is None:
            return out
        for key, col in (("models", "model"), ("harnesses", "harness"), ("runs", "run_id")):
            sql = f"SELECT DISTINCT {col} FROM traces WHERE {col} IS NOT NULL AND {col} != '' ORDER BY {col}"
            rows = conn.execute(sql).fetchall()
            out[key] = [r[0] for r in rows]
    return out


def get_traces(
    db_path: str | Path,
    session_id: str,
    *,
    since: float | None = None,
    limit: int | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Per-session ascending-by-time traces. Back-compat wrapper.

    New callers should prefer :func:`query_traces` directly.
    ``since`` here is strictly-newer-than (live-tail cursor).
    """
    return query_traces(
        db_path,
        session_id=session_id,
        run_id=run_id,
        since=since,
        limit=limit,
        order="ASC",
    )


# ---------------------------------------------------------------------------
# Session summaries (used by the Runs panel until Phase 3 lands)
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
        sql = "SELECT session_id FROM traces WHERE session_id != ''"
        params: list[Any] = []
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " GROUP BY session_id ORDER BY MIN(created_at) ASC"
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
    into one row.
    """
    with _connect(db_path) as conn:
        if conn is None:
            return {}
        sql = """
            SELECT session_id,
                   COUNT(*)        AS trace_count,
                   MIN(created_at) AS first_at,
                   MAX(created_at) AS last_at
            FROM traces
            WHERE session_id != ''
        """
        params: list[Any] = []
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " GROUP BY session_id"
        rows = conn.execute(sql, params).fetchall()
        return {
            r["session_id"]: {
                "trace_count": r["trace_count"],
                "first_at": r["first_at"],
                "last_at": r["last_at"],
            }
            for r in rows
        }


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

    Returns ``[]`` if the db doesn't have a ``runs`` table.
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

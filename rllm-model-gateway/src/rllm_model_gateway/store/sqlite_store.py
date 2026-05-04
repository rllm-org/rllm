"""SQLite-backed trace store with session-indexed persistence.

Schema v2 denormalizes a few hot fields from the JSON ``data`` column
onto the ``traces`` row (``run_id``, ``session_id``, ``model``,
``harness``, ``latency_ms``, ``has_error``, ``step_id``) so the global
trace feed can filter without table-scanning JSON. ``data`` keeps the
authoritative ``TraceRecord`` payload.

A schema-version mismatch on an existing db drops every table and
recreates from scratch — gateway data is dev-only and the cost of a
real migration isn't worth it. Bump :data:`_SCHEMA_VERSION` for any
schema change.
"""

import json
import logging
import os
import tempfile
import time
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# Bump on any schema change. ``_get_conn`` drops and recreates every
# table when the on-disk PRAGMA user_version is non-zero and ≠ this.
_SCHEMA_VERSION = 2

# Finish reasons that indicate a healthy LLM completion. Anything else
# (or a missing finish_reason on a non-empty response) gets has_error=1.
_OK_FINISH_REASONS = frozenset({"stop", "tool_calls", "length", "function_call"})


def _compute_has_error(data: dict[str, Any]) -> int:
    """Heuristic error flag derived from the trace payload at insert time.

    A trace counts as errored when:
    - ``raw_response`` carries an ``error`` key (upstream returned 4xx/5xx
      with an error body the gateway captured anyway), or
    - ``finish_reason`` is missing AND ``response_message`` is empty
      (the upstream returned no usable completion), or
    - ``finish_reason`` is set to something outside the well-known
      "good" set (e.g. ``content_filter``).
    """
    raw_response = data.get("raw_response")
    if isinstance(raw_response, dict) and "error" in raw_response:
        return 1
    finish_reason = data.get("finish_reason")
    response_message = data.get("response_message") or {}
    if not finish_reason:
        return 1 if not response_message else 0
    return 0 if finish_reason in _OK_FINISH_REASONS else 1


class SqliteTraceStore:
    """Persistent trace store backed by a single SQLite file.

    Uses a single persistent connection for the lifetime of the store
    (single-process gateway).  The connection is opened lazily on first
    use and closed explicitly via :meth:`close`.

    Features:
    - Denormalized filter columns on ``traces`` for the global feed
    - Junction table ``trace_sessions`` retained for many-session-per-trace
      back-compat (in practice 1:1)
    - WAL journal mode for faster local-disk read/write concurrency
    - Checkpoint-on-close to keep WAL growth bounded after long runs
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_dir = os.path.expanduser("~/.rllm")
            if not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True)
                except (OSError, PermissionError):
                    db_dir = tempfile.gettempdir()
            db_path = os.path.join(db_dir, "gateway_traces.db")
        else:
            db_path = os.path.expanduser(db_path)
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True)
                except (OSError, PermissionError) as exc:
                    logger.warning("Failed to create directory %s: %s", db_dir, exc)

        self.db_path = db_path
        self._busy_timeout_ms = 20_000
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _get_conn(self) -> aiosqlite.Connection:
        """Return the persistent connection, initializing on first call."""
        if self._conn is None:
            conn = await aiosqlite.connect(self.db_path, timeout=self._busy_timeout_ms / 1000.0)
            for pragma in (
                "PRAGMA journal_mode=WAL",
                "PRAGMA synchronous=NORMAL",
                f"PRAGMA busy_timeout={self._busy_timeout_ms}",
                "PRAGMA temp_store=MEMORY",
                "PRAGMA mmap_size=0",
            ):
                try:
                    await conn.execute(pragma)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SQLite pragma failed (%s): %s", pragma, exc)

            await conn.execute("PRAGMA foreign_keys = ON")
            await self._ensure_schema(conn)
            self._conn = conn
        return self._conn

    async def _ensure_schema(self, conn: aiosqlite.Connection) -> None:
        """Create or rebuild tables to match :data:`_SCHEMA_VERSION`.

        Three states to handle:
        - Fresh db (no tables, ``user_version=0``): just create v2.
        - v2 db (``user_version==_SCHEMA_VERSION``): no-op create-if-not-exists.
        - Anything else (legacy db with tables but no version stamp,
          or a future-version db downgraded): drop and recreate.
        """
        async with conn.execute("PRAGMA user_version") as cur:
            row = await cur.fetchone()
            current = int(row[0]) if row else 0

        async with conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('traces', 'trace_sessions', 'runs')") as cur:
            existing_tables = [r[0] async for r in cur]

        if existing_tables and current != _SCHEMA_VERSION:
            logger.warning(
                "rllm-gateway: dropping trace store at %s (schema v%d → v%d)",
                self.db_path,
                current,
                _SCHEMA_VERSION,
            )
            # Drop in dependency order: trace_sessions has an FK on traces.
            for table in ("trace_sessions", "traces", "runs"):
                await conn.execute(f"DROP TABLE IF EXISTS {table}")

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                trace_id    TEXT PRIMARY KEY,
                data        TEXT NOT NULL,
                created_at  REAL NOT NULL,
                session_id  TEXT NOT NULL DEFAULT '',
                run_id      TEXT NOT NULL DEFAULT '',
                model       TEXT NOT NULL DEFAULT '',
                harness     TEXT,
                latency_ms  INTEGER NOT NULL DEFAULT 0,
                has_error   INTEGER NOT NULL DEFAULT 0,
                step_id     INTEGER
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trace_sessions (
                trace_id   TEXT NOT NULL,
                session_id TEXT NOT NULL,
                run_id     TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                PRIMARY KEY (trace_id, session_id),
                FOREIGN KEY (trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id     TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                ended_at   REAL,
                metadata   TEXT NOT NULL DEFAULT '{}'
            )
            """
        )

        # Filter-pushdown indexes for the global trace feed. All on
        # ``traces`` because the feed query never needs the junction.
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_time ON traces(created_at DESC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_run_time ON traces(run_id, created_at DESC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_model_time ON traces(model, created_at DESC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_run_model_time ON traces(run_id, model, created_at DESC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_session_time ON traces(session_id, created_at DESC)")
        # Junction-side indexes — kept so legacy ``get_session_traces``
        # still hits an index. Phase 2/3 rewrites query ``traces``
        # directly via the new column.
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_session ON trace_sessions(session_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_session_time ON trace_sessions(session_id, created_at ASC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_run_session ON trace_sessions(run_id, session_id, created_at ASC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC)")

        await conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        await conn.commit()

    async def close(self) -> None:
        """Close the persistent connection and checkpoint WAL state."""
        if self._conn is not None:
            await self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            await self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # TraceStore protocol
    # ------------------------------------------------------------------

    async def store_trace(
        self,
        trace_id: str,
        session_id: str,
        data: dict[str, Any],
        run_id: str = "",
    ) -> None:
        conn = await self._get_conn()
        now = time.time()
        run_id = run_id or ""
        # Pull denormalized values from the TraceRecord-shaped dict. The
        # junction table's ``run_id`` is the authoritative gateway run
        # tag (set by the proxy at persist time), which we mirror onto
        # ``traces.run_id`` rather than trusting ``data["run_id"]``.
        model = str(data.get("model") or "")
        harness = data.get("harness")
        latency_ms = int(data.get("latency_ms") or 0)
        step_id = data.get("step_id")
        has_error = _compute_has_error(data)

        await conn.execute(
            """
            INSERT INTO traces (
                trace_id, data, created_at,
                session_id, run_id, model, harness, latency_ms, has_error, step_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trace_id) DO UPDATE SET
                data       = excluded.data,
                session_id = excluded.session_id,
                run_id     = excluded.run_id,
                model      = excluded.model,
                harness    = excluded.harness,
                latency_ms = excluded.latency_ms,
                has_error  = excluded.has_error,
                step_id    = excluded.step_id
            """,
            (
                trace_id,
                json.dumps(data),
                now,
                session_id,
                run_id,
                model,
                harness,
                latency_ms,
                has_error,
                step_id,
            ),
        )
        await conn.execute(
            """
            INSERT OR IGNORE INTO trace_sessions (trace_id, session_id, run_id, created_at)
            VALUES (?, ?, ?, COALESCE(
                (SELECT created_at FROM traces WHERE trace_id = ?), ?
            ))
            """,
            (trace_id, session_id, run_id, trace_id, now),
        )
        await conn.commit()

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        conn = await self._get_conn()
        conn.row_factory = aiosqlite.Row
        async with conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)) as cur:
            row = await cur.fetchone()
            if row is None:
                return None
            return json.loads(row["data"])

    async def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
        *,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get traces for a session, ordered ASC. Back-compat wrapper.

        New callers should prefer :meth:`query_traces` for richer
        filtering. ``run_id=None`` is cross-run.
        """
        conn = await self._get_conn()
        conn.row_factory = aiosqlite.Row
        sql = """
            SELECT t.data FROM traces t
            INNER JOIN trace_sessions ts ON t.trace_id = ts.trace_id
            WHERE ts.session_id = ?
        """
        params: list[Any] = [session_id]
        if run_id is not None:
            sql += " AND ts.run_id = ?"
            params.append(run_id)
        if since is not None:
            sql += " AND ts.created_at >= ?"
            params.append(since)
        sql += " ORDER BY ts.created_at ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        async with conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [json.loads(r["data"]) for r in rows]

    async def query_traces(
        self,
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
        """Filter+paginate traces directly off the denormalized columns.

        ``since`` / ``until`` are cursors on ``created_at``; pass
        ``until=last_seen`` for older-than-N scroll-back, ``since=last_seen``
        for newer-than-N live tail. ``order`` is ``DESC`` (newest first)
        for the global feed and ``ASC`` for per-session timelines.

        Each row carries a synthetic ``_created_at`` so callers can
        advance a polling cursor without inspecting ``timestamp`` inside
        ``data`` (which is the LLM request time and skews on long
        requests).
        """
        if order not in ("ASC", "DESC"):
            raise ValueError(f"order must be ASC or DESC, got {order!r}")
        conn = await self._get_conn()
        conn.row_factory = aiosqlite.Row

        where, params = self._build_filters(
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

        async with conn.execute(sql, params) as cur:
            rows = await cur.fetchall()

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

    async def count_traces(
        self,
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
    ) -> int:
        """COUNT(*) over the same filters as :meth:`query_traces`."""
        conn = await self._get_conn()
        where, params = self._build_filters(
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
        async with conn.execute(sql, params) as cur:
            row = await cur.fetchone()
        return int(row[0] or 0) if row else 0

    async def facets(self) -> dict[str, list[str]]:
        """Distinct values for filter-bar dropdowns.

        Returns ``{"models": [...], "harnesses": [...], "runs": [...]}``.
        Excludes the empty bucket from each list. Cheap because each
        column is indexed and the cardinality is small (handful of
        models/harnesses, dozens-to-hundreds of runs).
        """
        conn = await self._get_conn()

        async def _distinct(col: str, table: str = "traces") -> list[str]:
            sql = f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL AND {col} != '' ORDER BY {col}"
            async with conn.execute(sql) as cur:
                rows = await cur.fetchall()
            return [r[0] for r in rows]

        return {
            "models": await _distinct("model"),
            "harnesses": await _distinct("harness"),
            "runs": await _distinct("run_id"),
        }

    @staticmethod
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

    async def delete_session(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
    ) -> int:
        """Delete a session.

        ``run_id=None`` deletes every (session_id, *) pair across runs.
        Pass an explicit run_id to scope the delete to one run.
        """
        conn = await self._get_conn()
        # Find trace_ids unique to the (session_id [, run_id]) selector —
        # i.e. not referenced from any *other* row.
        scope_clause = "ts1.session_id = ?"
        scope_params: list[Any] = [session_id]
        anti_scope_clause = "ts2.session_id != ?"
        anti_scope_params: list[Any] = [session_id]
        if run_id is not None:
            scope_clause += " AND ts1.run_id = ?"
            scope_params.append(run_id)
            anti_scope_clause = "(ts2.session_id != ? OR ts2.run_id != ?)"
            anti_scope_params = [session_id, run_id]

        sql = f"""
            SELECT ts1.trace_id FROM trace_sessions ts1
            WHERE {scope_clause}
            AND NOT EXISTS (
                SELECT 1 FROM trace_sessions ts2
                WHERE ts2.trace_id = ts1.trace_id AND {anti_scope_clause}
            )
        """
        async with conn.execute(sql, [*scope_params, *anti_scope_params]) as cur:
            unique_rows = await cur.fetchall()
        unique_ids = [r[0] for r in unique_rows]

        # Delete junction rows for this scope
        if run_id is None:
            await conn.execute("DELETE FROM trace_sessions WHERE session_id = ?", (session_id,))
        else:
            await conn.execute(
                "DELETE FROM trace_sessions WHERE session_id = ? AND run_id = ?",
                (session_id, run_id),
            )
        # Delete orphaned traces
        if unique_ids:
            placeholders = ",".join("?" * len(unique_ids))
            await conn.execute(
                f"DELETE FROM traces WHERE trace_id IN ({placeholders})",
                unique_ids,
            )
        await conn.commit()
        return len(unique_ids)

    async def list_sessions(
        self,
        since: float | None = None,
        limit: int | None = None,
        *,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List sessions with trace counts.

        ``run_id=None`` is cross-run; rows include ``run_id`` so the
        caller can disambiguate. With an explicit ``run_id`` the result
        is filtered to one run.
        """
        conn = await self._get_conn()
        sql = """
            SELECT session_id,
                   run_id,
                   COUNT(*) as trace_count,
                   MIN(created_at) as first_trace_at,
                   MAX(created_at) as last_trace_at
            FROM trace_sessions
        """
        params: list[Any] = []
        where: list[str] = []
        if run_id is not None:
            where.append("run_id = ?")
            params.append(run_id)
        if since is not None:
            where.append("created_at >= ?")
            params.append(since)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY session_id, run_id ORDER BY MIN(created_at) DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        async with conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [
            {
                "session_id": r[0],
                "run_id": r[1],
                "trace_count": r[2],
                "first_trace_at": r[3],
                "last_trace_at": r[4],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Run-level metadata
    # ------------------------------------------------------------------

    async def register_run(self, run_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Record (or update) a gateway run's metadata.

        Idempotent: a second call with the same ``run_id`` updates the
        metadata in place but preserves ``started_at``. ``ended_at`` is
        cleared so a re-registered run reads as live again.
        """
        conn = await self._get_conn()
        now = time.time()
        meta_json = json.dumps(metadata or {})
        await conn.execute(
            """
            INSERT INTO runs (run_id, started_at, ended_at, metadata)
            VALUES (?, ?, NULL, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                metadata = excluded.metadata,
                ended_at = NULL
            """,
            (run_id, now, meta_json),
        )
        await conn.commit()

    async def end_run(self, run_id: str) -> None:
        """Stamp a run's ``ended_at`` with the current time."""
        conn = await self._get_conn()
        now = time.time()
        await conn.execute(
            "UPDATE runs SET ended_at = ? WHERE run_id = ? AND ended_at IS NULL",
            (now, run_id),
        )
        await conn.commit()

    async def list_runs(
        self,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List gateway runs ordered by ``started_at`` DESC."""
        conn = await self._get_conn()
        sql = "SELECT run_id, started_at, ended_at, metadata FROM runs"
        params: list[Any] = []
        if since is not None:
            sql += " WHERE started_at >= ?"
            params.append(since)
        sql += " ORDER BY started_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        async with conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                meta = json.loads(r[3]) if r[3] else {}
            except json.JSONDecodeError:
                meta = {}
            out.append(
                {
                    "run_id": r[0],
                    "started_at": r[1],
                    "ended_at": r[2],
                    "metadata": meta,
                }
            )
        return out

    async def flush(self) -> None:
        """No-op for SQLite (writes are synchronous within transactions)."""

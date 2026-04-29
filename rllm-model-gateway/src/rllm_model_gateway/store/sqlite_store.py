"""SQLite-backed trace store with session-indexed persistence.

Extracted and adapted from ``rllm/sdk/store/sqlite_store.py``.  Uses the same
junction-table pattern (``trace_sessions``) for efficient session-based
queries but simplifies the schema to match the gateway's needs.
"""

import json
import logging
import os
import tempfile
import time
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class SqliteTraceStore:
    """Persistent trace store backed by a single SQLite file.

    Uses a single persistent connection for the lifetime of the store
    (single-process gateway).  The connection is opened lazily on first
    use and closed explicitly via :meth:`close`.

    Features:
    - Junction table for session_id ↔ trace_id mapping
    - Composite indexes for fast session-scoped queries
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
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id   TEXT PRIMARY KEY,
                    data       TEXT NOT NULL,
                    created_at REAL NOT NULL
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
            # Lightweight forward migration: dbs created before run_id was a
            # column get the column added with the default. PK isn't widened
            # because trace_id is already a UUID — (trace_id, session_id)
            # remains unique; run_id is filterable metadata.
            async with conn.execute("PRAGMA table_info(trace_sessions)") as cur:
                cols = {row[1] async for row in cur}
            if "run_id" not in cols:
                await conn.execute("ALTER TABLE trace_sessions ADD COLUMN run_id TEXT NOT NULL DEFAULT ''")

            # ``runs`` records gateway-run metadata so the cross-run viewer
            # can list runs even before any traces have landed. Inserted by
            # the gateway lifespan via ``register_run``; ``ended_at`` is
            # filled in on shutdown.
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

            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_session ON trace_sessions(session_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_session_time ON trace_sessions(session_id, created_at ASC)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_run_session ON trace_sessions(run_id, session_id, created_at ASC)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_time ON traces(created_at)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC)")
            await conn.commit()
            self._conn = conn
        return self._conn

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
        await conn.execute(
            """
            INSERT OR REPLACE INTO traces (trace_id, data, created_at)
            VALUES (?, ?, COALESCE((SELECT created_at FROM traces WHERE trace_id = ?), ?))
            """,
            (trace_id, json.dumps(data), trace_id, now),
        )
        await conn.execute(
            """
            INSERT OR IGNORE INTO trace_sessions (trace_id, session_id, run_id, created_at)
            VALUES (?, ?, ?, COALESCE(
                (SELECT created_at FROM traces WHERE trace_id = ?), ?
            ))
            """,
            (trace_id, session_id, run_id or "", trace_id, now),
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
        """Get traces for a session, optionally narrowed to one ``run_id``.

        ``run_id=None`` (default) is cross-run — useful when an external
        client used the same ``session_id`` across multiple gateway runs.
        Pass an empty string to match the unstamped bucket explicitly.
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

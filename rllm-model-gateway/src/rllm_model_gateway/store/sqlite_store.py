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
    - DELETE journal mode for NFS compatibility (not WAL)
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
                "PRAGMA journal_mode=DELETE",
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
                    created_at REAL NOT NULL,
                    PRIMARY KEY (trace_id, session_id),
                    FOREIGN KEY (trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
                )
                """
            )
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_session ON trace_sessions(session_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_session_time ON trace_sessions(session_id, created_at ASC)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_time ON traces(created_at)")
            await conn.commit()
            self._conn = conn
        return self._conn

    async def close(self) -> None:
        """Close the persistent connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # TraceStore protocol
    # ------------------------------------------------------------------

    async def store_trace(self, trace_id: str, session_id: str, data: dict[str, Any]) -> None:
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
            INSERT OR IGNORE INTO trace_sessions (trace_id, session_id, created_at)
            VALUES (?, ?, COALESCE(
                (SELECT created_at FROM traces WHERE trace_id = ?), ?
            ))
            """,
            (trace_id, session_id, trace_id, now),
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
    ) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        conn.row_factory = aiosqlite.Row
        sql = """
            SELECT t.data FROM traces t
            INNER JOIN trace_sessions ts ON t.trace_id = ts.trace_id
            WHERE ts.session_id = ?
        """
        params: list[Any] = [session_id]
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

    async def delete_session(self, session_id: str) -> int:
        conn = await self._get_conn()
        # Find trace_ids unique to this session (not shared with others)
        async with conn.execute(
            """
            SELECT ts1.trace_id FROM trace_sessions ts1
            WHERE ts1.session_id = ?
            AND NOT EXISTS (
                SELECT 1 FROM trace_sessions ts2
                WHERE ts2.trace_id = ts1.trace_id AND ts2.session_id != ?
            )
            """,
            (session_id, session_id),
        ) as cur:
            unique_rows = await cur.fetchall()
        unique_ids = [r[0] for r in unique_rows]

        # Delete junction rows for this session
        await conn.execute("DELETE FROM trace_sessions WHERE session_id = ?", (session_id,))
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
    ) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        sql = """
            SELECT session_id,
                   COUNT(*) as trace_count,
                   MIN(created_at) as first_trace_at,
                   MAX(created_at) as last_trace_at
            FROM trace_sessions
        """
        params: list[Any] = []
        if since is not None:
            sql += " WHERE created_at >= ?"
            params.append(since)
        sql += " GROUP BY session_id ORDER BY MIN(created_at) DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        async with conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [
            {
                "session_id": r[0],
                "trace_count": r[1],
                "first_trace_at": r[2],
                "last_trace_at": r[3],
            }
            for r in rows
        ]

    async def flush(self) -> None:
        """No-op for SQLite (writes are synchronous within transactions)."""

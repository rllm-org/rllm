"""Standalone SQLite trace store with session context tracking."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class TraceContext:
    """
    Simple context object returned by the store.

    Contains only essential fields for trace/signal storage and retrieval.
    """

    def __init__(
        self,
        id: str,
        data: dict[str, Any],
        namespace: str = "default",
        type: str = "generic",
        metadata: dict[str, Any] | None = None,
        created_at: float | None = None,
        updated_at: float | None = None,
    ):
        self.id = id
        self.data = data
        self.namespace = namespace
        self.type = type
        self.metadata = metadata or {}
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or time.time()


class SqliteTraceStore:
    """
    Standalone SQLite-based trace store with fast session context queries.

    Uses a junction table pattern to enable efficient queries by session_uid
    while avoiding data duplication. Each trace is stored once, but can be associated
    with multiple session contexts (for nested sessions).

    This store does NOT perform any auto-detection of session context.
    Callers must explicitly provide session_uids when storing traces.

    Features:
    - Fast queries by session_uid (indexed junction table)
    - Single table for traces and signals (differentiated by context_type)
    - Batch insert support with transactions
    - DELETE journal mode for compatibility (not WAL)
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize SQLite trace store.

        Args:
            db_path: Path to SQLite database file. If None, uses default location
                    (~/.rllm/traces.db or temp directory)
        """
        if db_path is None:
            db_dir = os.path.expanduser("~/.rllm")
            if not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True)
                except (OSError, PermissionError):
                    db_dir = tempfile.gettempdir()
            db_path = os.path.join(db_dir, "traces.db")
        else:
            # Expand ~ and ensure parent directory exists
            db_path = os.path.expanduser(db_path)
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True)
                except (OSError, PermissionError) as e:
                    logger.warning(f"Failed to create directory {db_dir}: {e}")

        self.db_path = db_path
        self._sqlite_busy_timeout_ms = 20000
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure database is initialized (lazy initialization)."""
        if not self._initialized:
            await self._init_database()
            self._initialized = True

    async def _configure_connection(self, conn: aiosqlite.Connection) -> None:
        """Configure SQLite connection with pragmas for compatibility."""
        pragmas = [
            "PRAGMA journal_mode=DELETE",  # Use DELETE mode, not WAL (better compatibility)
            "PRAGMA synchronous=NORMAL",
            f"PRAGMA busy_timeout={self._sqlite_busy_timeout_ms}",
            "PRAGMA temp_store=MEMORY",
            "PRAGMA mmap_size=0",  # Disable mmap for network FS compatibility
        ]
        for pragma in pragmas:
            try:
                await conn.execute(pragma)
            except Exception as exc:
                logger.warning("SQLite pragma failed (%s): %s", pragma, exc)

    async def _connect(self) -> aiosqlite.Connection:
        """Create a configured async SQLite connection."""
        conn = await aiosqlite.connect(self.db_path, timeout=self._sqlite_busy_timeout_ms / 1000.0)
        await self._configure_connection(conn)
        return conn

    async def _init_database(self) -> None:
        """Initialize the SQLite database with required schema."""
        conn = await self._connect()
        try:
            await conn.execute("PRAGMA foreign_keys = ON")

            # Create traces table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    context_type TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    data TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

            # Create junction table for session context mapping
            # Note: created_at is denormalized here for query performance
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trace_sessions (
                    trace_id TEXT NOT NULL,
                    session_uid TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (trace_id, session_uid),
                    FOREIGN KEY (trace_id) REFERENCES traces(id) ON DELETE CASCADE
                )
            """)

            # Create indexes for performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_sessions_uid ON trace_sessions(session_uid)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_sessions_trace ON trace_sessions(trace_id)")
            # Composite index for efficient time-bounded session queries (created after migration)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_sessions_uid_time ON trace_sessions(session_uid, created_at DESC)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_type ON traces(context_type)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_namespace ON traces(namespace)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces(created_at)")

            await conn.commit()
        finally:
            await conn.close()

    async def store(
        self,
        trace_id: str,
        data: dict[str, Any],
        namespace: str = "default",
        context_type: str = "llm_trace",
        metadata: dict[str, Any] | None = None,
        session_uids: list[str] | None = None,
    ) -> None:
        """
        Store a trace or signal.

        Creates junction table entries for fast queries by session_uid.
        The store does NOT perform any auto-detection - callers must provide
        all values including session_uids.

        Args:
            trace_id: Unique trace/signal ID
            data: Trace/signal data (dict)
            namespace: Namespace for organization (default: "default")
            context_type: Type of context ('llm_trace' or signal type) (default: "llm_trace")
            metadata: Metadata dictionary (default: {})
            session_uids: List of session UIDs to associate with this trace (caller must provide)
        """
        await self._ensure_initialized()
        now = time.time()

        conn = await self._connect()
        try:
            # Insert or replace trace
            await conn.execute(
                """
                INSERT OR REPLACE INTO traces
                (id, context_type, namespace, data, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?,
                    COALESCE((SELECT created_at FROM traces WHERE id = ?), ?),
                    ?)
                """,
                (
                    trace_id,
                    context_type,
                    namespace,
                    json.dumps(data),
                    json.dumps(metadata),
                    trace_id,  # For SELECT in COALESCE
                    now,  # created_at if new
                    now,  # updated_at
                ),
            )

            # Read back the actual created_at that was persisted (may differ from 'now' if updated)
            async with conn.execute("SELECT created_at FROM traces WHERE id = ?", (trace_id,)) as cursor:
                row = await cursor.fetchone()
                actual_created_at = row[0] if row else now

            # Delete existing junction entries for this trace
            await conn.execute("DELETE FROM trace_sessions WHERE trace_id = ?", (trace_id,))

            # Insert new junction entries (one per active session context)
            # This creates the many-to-many mapping: one trace → multiple session contexts
            # Example: If trace is logged in outer + inner session:
            #   - trace_id="tr_123" → session_uid="ctx_outer" (row 1)
            #   - trace_id="tr_123" → session_uid="ctx_inner" (row 2)
            # The composite index on (session_uid, created_at) enables fast time-bounded queries
            # CRITICAL: Use actual_created_at (not 'now') to keep junction table in sync with traces table
            if session_uids:
                await conn.executemany(
                    "INSERT INTO trace_sessions (trace_id, session_uid, created_at) VALUES (?, ?, ?)",
                    [(trace_id, uid, actual_created_at) for uid in session_uids],
                )

            await conn.commit()
        finally:
            await conn.close()

    async def store_batch(
        self,
        traces: list[dict[str, Any]],
        preserve_order: bool = True,
        session_uids: list[str] | None = None,
    ) -> list[TraceContext]:
        """
        Store multiple traces/signals in a batch.

        The store does NOT perform any auto-detection - callers must provide
        all values including session_uids.

        Args:
            traces: List of trace dicts with keys:
                   - id (trace_id)
                   - data
                   - namespace (optional, default: "default")
                   - type (context_type, optional, default: "llm_trace")
                   - metadata (optional, default: {})
                   - session_uids (optional, overrides the batch-level session_uids)
            preserve_order: If True, maintains insertion order (default: True)
            session_uids: Default session UIDs for all traces (caller must provide)

        Returns:
            List of stored TraceContext objects
        """
        await self._ensure_initialized()
        now = time.time()
        stored = []

        conn = await self._connect()
        try:
            for trace in traces:
                trace_id = trace["id"]
                data = trace["data"]
                namespace = trace.get("namespace", "default")
                context_type = trace.get("type", "llm_trace")
                metadata = trace.get("metadata", {})
                # Per-trace session_uids overrides batch-level session_uids
                trace_session_uids = trace.get("session_uids", session_uids)

                # Insert or replace trace
                await conn.execute(
                    """
                    INSERT OR REPLACE INTO traces
                    (id, context_type, namespace, data, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?,
                        COALESCE((SELECT created_at FROM traces WHERE id = ?), ?),
                        ?)
                    """,
                    (
                        trace_id,
                        context_type,
                        namespace,
                        json.dumps(data),
                        json.dumps(metadata),
                        trace_id,
                        now,
                        now,
                    ),
                )

                # Read back the actual created_at that was persisted (may differ from 'now' if updated)
                async with conn.execute("SELECT created_at FROM traces WHERE id = ?", (trace_id,)) as cursor:
                    row = await cursor.fetchone()
                    actual_created_at = row[0] if row else now

                # Delete existing junction entries
                await conn.execute("DELETE FROM trace_sessions WHERE trace_id = ?", (trace_id,))

                # Insert new junction entries
                # CRITICAL: Use actual_created_at (not 'now') to keep junction table in sync with traces table
                if trace_session_uids:
                    await conn.executemany(
                        "INSERT INTO trace_sessions (trace_id, session_uid, created_at) VALUES (?, ?, ?)",
                        [(trace_id, uid, actual_created_at) for uid in trace_session_uids],
                    )

                # Create TraceContext for return
                stored.append(
                    TraceContext(
                        id=trace_id,
                        data=data,
                        namespace=namespace,
                        type=context_type,
                        metadata=metadata,
                        created_at=actual_created_at,
                        updated_at=now,
                    )
                )

            await conn.commit()
        finally:
            await conn.close()

        return stored

    async def get(self, trace_id: str) -> TraceContext | None:
        """
        Get a trace/signal by ID.

        Args:
            trace_id: Trace/signal ID

        Returns:
            TraceContext object or None if not found
        """
        await self._ensure_initialized()
        conn = await self._connect()
        try:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)) as cursor:
                row = await cursor.fetchone()

                if not row:
                    return None

                return TraceContext(
                    id=row["id"],
                    data=json.loads(row["data"]),
                    namespace=row["namespace"],
                    type=row["context_type"],
                    metadata=json.loads(row["metadata"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
        finally:
            await conn.close()

    async def query(
        self,
        session_uids: list[str] | None = None,
        context_types: list[str] | None = None,
        namespaces: list[str] | None = None,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[TraceContext]:
        """
        Query traces/signals with filters.

        Args:
            session_uids: Filter by session UIDs (list of session UIDs)
            context_types: Filter by context types (e.g., ['llm_trace'])
            namespaces: Filter by namespaces
            since: Filter by created_at >= since (Unix timestamp). Early filter for performance.
            limit: Maximum number of results to return

        Returns:
            List of TraceContext objects matching the filters
        """

        await self._ensure_initialized()
        conn = await self._connect()
        try:
            conn.row_factory = aiosqlite.Row

            # Build query
            query_parts = ["SELECT DISTINCT t.* FROM traces t"]
            params = []

            # Join with junction table if filtering by session_uid (must come before WHERE)
            if session_uids:
                query_parts.append("JOIN trace_sessions ts ON t.id = ts.trace_id")

            # Start WHERE clause
            where_conditions = []

            # Apply since filter early for performance
            # Key optimization: When filtering by session_uids, use ts.created_at
            # to leverage the composite index (session_uid, created_at)
            if since is not None:
                if session_uids:
                    where_conditions.append("ts.created_at >= ?")
                else:
                    where_conditions.append("t.created_at >= ?")
                params.append(since)

            # Add session_uid filter
            if session_uids:
                placeholders = ",".join(["?"] * len(session_uids))
                where_conditions.append(f"ts.session_uid IN ({placeholders})")
                params.extend(session_uids)

            # Add other filters
            if context_types:
                placeholders = ",".join(["?"] * len(context_types))
                where_conditions.append(f"t.context_type IN ({placeholders})")
                params.extend(context_types)

            if namespaces:
                placeholders = ",".join(["?"] * len(namespaces))
                where_conditions.append(f"t.namespace IN ({placeholders})")
                params.extend(namespaces)

            # Add WHERE clause if there are any conditions
            if where_conditions:
                query_parts.append("WHERE " + " AND ".join(where_conditions))

            # Order by created_at descending
            # Note: We filter by ts.created_at (to use composite index) but order by t.created_at
            # (to satisfy DISTINCT requirement). Since ts.created_at is denormalized from
            # t.created_at, they are equal and the ordering is semantically equivalent.
            query_parts.append("ORDER BY t.created_at DESC")

            # Limit
            if limit:
                query_parts.append("LIMIT ?")
                params.append(limit)

            query = " ".join(query_parts)

            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()

                results = []
                for row in rows:
                    results.append(
                        TraceContext(
                            id=row["id"],
                            data=json.loads(row["data"]),
                            namespace=row["namespace"],
                            type=row["context_type"],
                            metadata=json.loads(row["metadata"]),
                            created_at=row["created_at"],
                            updated_at=row["updated_at"],
                        )
                    )

                return results
        finally:
            await conn.close()

    async def get_session_uids_for_trace(self, trace_id: str) -> list[str]:
        """
        Get all session UIDs that a trace belongs to.

        Useful for understanding which nested sessions a trace was logged in.
        Shows the many-to-many relationship: one trace can belong to multiple session contexts.

        Args:
            trace_id: Trace ID to look up

        Returns:
            List of session_uids that this trace belongs to
        """
        await self._ensure_initialized()
        conn = await self._connect()
        try:
            async with conn.execute(
                "SELECT session_uid FROM trace_sessions WHERE trace_id = ?",
                (trace_id,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
        finally:
            await conn.close()

    async def get_by_session_uid(
        self,
        session_uid: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[TraceContext]:
        """
        Fast lookup of all traces for a session context.

        Uses the composite indexed junction table for optimal performance.

        Args:
            session_uid: Session UID to query
            since: Filter by created_at >= since (Unix timestamp). Early filter for performance.
            limit: Maximum number of results to return (most recent first)

        Returns:
            List of TraceContext objects for this session context
        """
        await self._ensure_initialized()
        conn = await self._connect()
        try:
            conn.row_factory = aiosqlite.Row

            # Optimized query using composite index (session_uid, created_at):
            # 1. Composite index (idx_trace_sessions_uid_time) → O(log n) lookup + range scan
            # 2. Filter on junction table's created_at (no need to join first)
            # 3. Join to traces only for matching rows → O(k) where k = result set size
            if since is not None:
                # Use composite index for both session_uid and time filtering
                query = """
                    SELECT t.* FROM traces t
                    INNER JOIN trace_sessions ts ON t.id = ts.trace_id
                    WHERE ts.session_uid = ? AND ts.created_at >= ?
                    ORDER BY ts.created_at DESC
                """
                params = [session_uid, since]
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)

                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
            else:
                # Fast lookup using session_uid index
                query = """
                    SELECT t.* FROM traces t
                    INNER JOIN trace_sessions ts ON t.id = ts.trace_id
                    WHERE ts.session_uid = ?
                    ORDER BY ts.created_at DESC
                """
                params = [session_uid]
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)

                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

            results = []
            for row in rows:
                results.append(
                    TraceContext(
                        id=row["id"],
                        data=json.loads(row["data"]),
                        namespace=row["namespace"],
                        type=row["context_type"],
                        metadata=json.loads(row["metadata"]),
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                )

            return results
        finally:
            await conn.close()

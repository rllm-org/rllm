"""SQLite-backed trace store. One file, three tables, WAL mode.

Schema:
  sessions      — one row per session (metadata + sampling_params)
  traces        — one row per request (typed metadata columns)
  trace_extras  — one row per request, blob (large training-side data)

The split between traces and trace_extras keeps the hot path
``get_traces(session_id)`` free of blob I/O.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any

import aiosqlite

from rllm_model_gateway.normalized import Message, ToolCall, ToolSpec, Usage
from rllm_model_gateway.trace import TraceRecord

logger = logging.getLogger(__name__)


def _to_json(obj: Any) -> str | None:
    """Pydantic-aware JSON serializer for column storage."""
    if obj is None:
        return None
    if isinstance(obj, list):
        return json.dumps(
            [item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item for item in obj],
            ensure_ascii=False,
        )
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(exclude_none=True), ensure_ascii=False)
    return json.dumps(obj, ensure_ascii=False)


def _from_json_list(s: str | None, model_cls) -> list | None:
    if not s:
        return None
    raw = json.loads(s)
    return [model_cls(**item) for item in raw]


def _from_json_dict(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    return json.loads(s)


class SqliteTraceStore:
    """Persistent trace store. Single connection, WAL mode."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_dir = os.path.expanduser("~/.rllm")
            try:
                os.makedirs(db_dir, exist_ok=True)
            except (OSError, PermissionError):
                db_dir = tempfile.gettempdir()
            db_path = os.path.join(db_dir, "gateway.db")
        else:
            db_path = os.path.expanduser(db_path)
            db_dir = os.path.dirname(db_path)
            if db_dir:
                try:
                    os.makedirs(db_dir, exist_ok=True)
                except (OSError, PermissionError) as exc:
                    logger.warning("Failed to create directory %s: %s", db_dir, exc)

        self.db_path = db_path
        self._busy_timeout_ms = 20_000
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is not None:
            return self._conn
        conn = await aiosqlite.connect(self.db_path, timeout=self._busy_timeout_ms / 1000.0)
        for pragma in (
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            f"PRAGMA busy_timeout={self._busy_timeout_ms}",
            "PRAGMA temp_store=MEMORY",
            "PRAGMA foreign_keys=ON",
        ):
            try:
                await conn.execute(pragma)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SQLite pragma failed (%s): %s", pragma, exc)

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id      TEXT PRIMARY KEY,
                created_at      REAL NOT NULL,
                metadata        TEXT,
                sampling_params TEXT
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                trace_id          TEXT PRIMARY KEY,
                session_id        TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                timestamp         REAL NOT NULL,
                endpoint          TEXT NOT NULL,
                model             TEXT,
                messages          TEXT,
                prompt            TEXT,
                tools             TEXT,
                kwargs            TEXT,
                content           TEXT,
                text              TEXT,
                reasoning         TEXT,
                tool_calls        TEXT,
                finish_reason     TEXT,
                prompt_tokens     INTEGER,
                completion_tokens INTEGER,
                metrics           TEXT,
                metadata          TEXT
            )
            """
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id, timestamp)")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trace_extras (
                trace_id  TEXT PRIMARY KEY REFERENCES traces(trace_id) ON DELETE CASCADE,
                format    TEXT NOT NULL,
                data      BLOB NOT NULL
            )
            """
        )
        await conn.commit()
        self._conn = conn
        return conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def flush(self) -> None:
        # WAL mode auto-commits per transaction; explicit commit is not needed
        # but issue PRAGMA wal_checkpoint to ensure durability if requested.
        if self._conn is not None:
            try:
                await self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def create_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> None:
        conn = await self._get_conn()
        await conn.execute(
            """
            INSERT INTO sessions (session_id, created_at, metadata, sampling_params)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                metadata = COALESCE(excluded.metadata, sessions.metadata),
                sampling_params = COALESCE(excluded.sampling_params, sessions.sampling_params)
            """,
            (
                session_id,
                time.time(),
                json.dumps(metadata) if metadata else None,
                json.dumps(sampling_params) if sampling_params else None,
            ),
        )
        await conn.commit()

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        conn = await self._get_conn()
        conn.row_factory = aiosqlite.Row
        async with conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        async with conn.execute("SELECT COUNT(*) FROM traces WHERE session_id = ?", (session_id,)) as cur:
            count_row = await cur.fetchone()
        return {
            "session_id": row["session_id"],
            "created_at": row["created_at"],
            "metadata": _from_json_dict(row["metadata"]),
            "sampling_params": _from_json_dict(row["sampling_params"]),
            "trace_count": count_row[0] if count_row else 0,
        }

    async def list_sessions(
        self,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        conn = await self._get_conn()
        conn.row_factory = aiosqlite.Row
        sql = """
            SELECT s.session_id, s.created_at, s.metadata, s.sampling_params,
                   (SELECT COUNT(*) FROM traces WHERE traces.session_id = s.session_id) AS trace_count
            FROM sessions s
        """
        params: list[Any] = []
        if since is not None:
            sql += " WHERE s.created_at >= ?"
            params.append(since)
        sql += " ORDER BY s.created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        async with conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [
            {
                "session_id": r["session_id"],
                "created_at": r["created_at"],
                "metadata": _from_json_dict(r["metadata"]),
                "sampling_params": _from_json_dict(r["sampling_params"]),
                "trace_count": r["trace_count"],
            }
            for r in rows
        ]

    async def delete_session(self, session_id: str) -> int:
        conn = await self._get_conn()
        async with conn.execute("SELECT COUNT(*) FROM traces WHERE session_id = ?", (session_id,)) as cur:
            row = await cur.fetchone()
        count = row[0] if row else 0
        # FK ON DELETE CASCADE handles traces and trace_extras
        await conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await conn.commit()
        return count

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    async def store_trace(
        self,
        trace: TraceRecord,
        extras: tuple[str, bytes] | None = None,
    ) -> None:
        conn = await self._get_conn()

        # Sessions must be created explicitly via create_session() before any
        # trace is stored. Trying to insert a trace for an unknown session
        # raises sqlite3.IntegrityError (FK violation), which the gateway's
        # request handler should prevent by checking existence first.
        await conn.execute(
            """
            INSERT OR REPLACE INTO traces (
                trace_id, session_id, timestamp, endpoint, model,
                messages, prompt, tools, kwargs,
                content, text, reasoning, tool_calls, finish_reason,
                prompt_tokens, completion_tokens, metrics, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.trace_id,
                trace.session_id,
                trace.timestamp,
                trace.endpoint,
                trace.model,
                _to_json(trace.messages),
                trace.prompt,
                _to_json(trace.tools),
                _to_json(trace.kwargs),
                trace.content,
                trace.text,
                trace.reasoning,
                _to_json(trace.tool_calls),
                trace.finish_reason,
                trace.usage.prompt_tokens,
                trace.usage.completion_tokens,
                json.dumps(trace.metrics) if trace.metrics else None,
                json.dumps(trace.metadata) if trace.metadata else None,
            ),
        )
        if extras is not None:
            fmt, data = extras
            await conn.execute(
                """
                INSERT OR REPLACE INTO trace_extras (trace_id, format, data)
                VALUES (?, ?, ?)
                """,
                (trace.trace_id, fmt, data),
            )
        await conn.commit()

    async def get_trace(self, trace_id: str, extras: bool = False) -> TraceRecord | None:
        conn = await self._get_conn()
        conn.row_factory = aiosqlite.Row
        if extras:
            sql = "SELECT t.*, x.format AS _extras_format, x.data AS _extras_data FROM traces t LEFT JOIN trace_extras x ON x.trace_id = t.trace_id WHERE t.trace_id = ?"
        else:
            sql = "SELECT * FROM traces WHERE trace_id = ?"
        async with conn.execute(sql, (trace_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_trace(row, with_extras=extras)

    async def get_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
        extras: bool = False,
    ) -> list[TraceRecord]:
        conn = await self._get_conn()
        conn.row_factory = aiosqlite.Row
        if extras:
            sql = "SELECT t.*, x.format AS _extras_format, x.data AS _extras_data FROM traces t LEFT JOIN trace_extras x ON x.trace_id = t.trace_id WHERE t.session_id = ?"
        else:
            sql = "SELECT * FROM traces WHERE session_id = ?"
        params: list[Any] = [session_id]
        ts_col = "t.timestamp" if extras else "timestamp"
        if since is not None:
            sql += f" AND {ts_col} >= ?"
            params.append(since)
        sql += f" ORDER BY {ts_col} ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        async with conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_trace(r, with_extras=extras) for r in rows]

    async def get_trace_extras(self, trace_id: str) -> tuple[str, bytes] | None:
        """Standalone fetch of an extras blob. Mostly useful for the
        HTTP endpoint; Python consumers should use ``get_trace(..., extras=True)``
        or ``get_traces(..., extras=True)``."""
        conn = await self._get_conn()
        conn.row_factory = aiosqlite.Row
        async with conn.execute("SELECT format, data FROM trace_extras WHERE trace_id = ?", (trace_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return (row["format"], row["data"])


def _row_to_trace(row, *, with_extras: bool = False) -> TraceRecord:
    extras: dict[str, Any] | None
    if with_extras:
        fmt = row["_extras_format"] if "_extras_format" in row.keys() else None
        data = row["_extras_data"] if "_extras_data" in row.keys() else None
        if fmt and data is not None:
            from rllm_model_gateway.trace import deserialize_extras

            extras = deserialize_extras(fmt, data)
        else:
            extras = {}
    else:
        extras = None
    return TraceRecord(
        trace_id=row["trace_id"],
        session_id=row["session_id"],
        timestamp=row["timestamp"],
        endpoint=row["endpoint"],
        model=row["model"] or "",
        messages=_from_json_list(row["messages"], Message),
        prompt=row["prompt"],
        tools=_from_json_list(row["tools"], ToolSpec),
        kwargs=_from_json_dict(row["kwargs"]),
        content=row["content"] or "",
        text=row["text"],
        reasoning=row["reasoning"],
        tool_calls=_from_json_list(row["tool_calls"], ToolCall) or [],
        finish_reason=row["finish_reason"] or "stop",
        usage=Usage(
            prompt_tokens=row["prompt_tokens"] or 0,
            completion_tokens=row["completion_tokens"] or 0,
        ),
        metrics=_from_json_dict(row["metrics"]),
        metadata=_from_json_dict(row["metadata"]),
        extras=extras,
    )

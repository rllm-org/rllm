"""SQLite-backed execution result store for sandbox workers.

Uses the same DB file as SqliteTraceStore. WAL mode enables concurrent
access from the trainer process (register/wait) and the proxy subprocess
(store_result via POST route).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import time

from rllm.sdk.sandbox.protocol import ExecutionResult
from rllm.sdk.sandbox.serialization import deserialize_execution_result

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS execution_results (
    execution_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    trajectories TEXT,
    session_uid TEXT,
    reward REAL,
    error TEXT,
    elapsed REAL DEFAULT 0.0,
    created_at REAL NOT NULL,
    completed_at REAL
);
"""

_BUSY_TIMEOUT_MS = 20_000


class ExecutionResultStore:
    """SQLite-backed store for execution results pushed by sandbox workers.

    Cross-process flow:
    - Trainer process calls ``register()`` and ``wait()``.
    - Proxy subprocess calls ``store_result()`` via the POST route.
    - WAL mode handles concurrent reads/writes from different processes.
    """

    def __init__(self, db_path: str | None = None, pool_size: int = 2):
        if db_path is None:
            db_dir = os.path.expanduser("~/.rllm")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "traces.db")
        else:
            db_path = os.path.expanduser(db_path)
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

        self.db_path = db_path
        self._pool_size = pool_size
        self._connections: list[sqlite3.Connection] = []
        self._init_database()

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_MS / 1000.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    def _init_database(self) -> None:
        conn = self._create_connection()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

        for _ in range(self._pool_size):
            self._connections.append(self._create_connection())

    def _conn(self) -> sqlite3.Connection:
        """Return a connection from the pool (round-robin)."""
        return self._connections[0] if self._connections else self._create_connection()

    def register(self, execution_id: str) -> None:
        """Register a pending execution (called by trainer before dispatch)."""
        conn = self._conn()
        conn.execute(
            "INSERT OR IGNORE INTO execution_results (execution_id, status, created_at) VALUES (?, 'pending', ?)",
            (execution_id, time.time()),
        )
        conn.commit()

    def store_result(self, execution_id: str, data: dict) -> None:
        """Store a completed result (called by proxy route from worker push)."""
        conn = self._conn()
        conn.execute(
            """UPDATE execution_results
               SET status = 'completed',
                   trajectories = ?,
                   session_uid = ?,
                   reward = ?,
                   error = ?,
                   elapsed = ?,
                   completed_at = ?
               WHERE execution_id = ?""",
            (
                json.dumps(data.get("trajectories")) if data.get("trajectories") is not None else None,
                data.get("session_uid", ""),
                data.get("reward"),
                data.get("error"),
                data.get("elapsed", 0.0),
                time.time(),
                execution_id,
            ),
        )
        # If the row didn't exist yet (race: worker faster than register), insert it
        if conn.total_changes == 0 or conn.execute("SELECT changes()").fetchone()[0] == 0:
            conn.execute(
                """INSERT OR REPLACE INTO execution_results
                   (execution_id, status, trajectories, session_uid, reward, error, elapsed, created_at, completed_at)
                   VALUES (?, 'completed', ?, ?, ?, ?, ?, ?, ?)""",
                (
                    execution_id,
                    json.dumps(data.get("trajectories")) if data.get("trajectories") is not None else None,
                    data.get("session_uid", ""),
                    data.get("reward"),
                    data.get("error"),
                    data.get("elapsed", 0.0),
                    time.time(),
                    time.time(),
                ),
            )
        conn.commit()

    def get_result(self, execution_id: str) -> ExecutionResult | None:
        """Get an execution result (returns None if pending or not found)."""
        conn = self._conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM execution_results WHERE execution_id = ?",
            (execution_id,),
        )
        row = cursor.fetchone()
        if row is None or row["status"] == "pending":
            return None

        trajectories = json.loads(row["trajectories"]) if row["trajectories"] else None
        return deserialize_execution_result({
            "success": row["error"] is None,
            "trajectories": trajectories,
            "session_uid": row["session_uid"] or "",
            "reward": row["reward"],
            "error": row["error"],
            "elapsed": row["elapsed"] or 0.0,
        })

    def wait(self, execution_id: str, timeout: float = 600.0, poll_interval: float = 1.0) -> ExecutionResult:
        """Poll until result is available or timeout expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self.get_result(execution_id)
            if result is not None:
                return result
            time.sleep(poll_interval)

        return ExecutionResult(
            success=False,
            error=f"Timed out waiting for execution {execution_id} after {timeout}s",
        )

    def close(self) -> None:
        """Close all pooled connections."""
        for conn in self._connections:
            try:
                conn.close()
            except Exception:
                pass
        self._connections.clear()

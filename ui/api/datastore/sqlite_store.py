import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .base import DataStore, extract_searchable_text


class SQLiteStore(DataStore):
    def __init__(self, db_path: str = "rllm_ui.db"):
        self.db_path = Path(__file__).parent.parent / db_path

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    experiment TEXT NOT NULL,
                    config JSON,
                    source_metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)

            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT REFERENCES sessions(id),
                    step INTEGER,
                    data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Episodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    session_id TEXT REFERENCES sessions(id),
                    step INTEGER,
                    task JSON,
                    is_correct BOOLEAN,
                    reward REAL,
                    data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Check for migrations (schema updates)
            # Migration: Add source_metadata column if it doesn't exist
            cursor.execute("PRAGMA table_info(sessions)")
            columns = [col[1] for col in cursor.fetchall()]
            if "source_metadata" not in columns:
                cursor.execute("ALTER TABLE sessions ADD COLUMN source_metadata JSON")

            # Migration: Add search_text column for full-text search
            cursor.execute("PRAGMA table_info(episodes)")
            episode_columns = [col[1] for col in cursor.fetchall()]
            if "search_text" not in episode_columns:
                cursor.execute("ALTER TABLE episodes ADD COLUMN search_text TEXT")

            conn.commit()

    def reset(self):
        if self.db_path.exists():
            self.db_path.unlink()
        self.init_db()

    def create_session(self, project: str, experiment: str, config: dict[str, Any], source_metadata: dict[str, Any]) -> str:
        session_id = str(uuid.uuid4())
        with self._get_conn() as conn:
            conn.execute("INSERT INTO sessions (id, project, experiment, config, source_metadata) VALUES (?, ?, ?, ?, ?)", (session_id, project, experiment, json.dumps(config), json.dumps(source_metadata)))
            conn.commit()
        return session_id

    def log_metrics(self, session_id: str, step: int, data: dict[str, Any]) -> dict[str, Any] | None:
        with self._get_conn() as conn:
            cursor = conn.execute("INSERT INTO metrics (session_id, step, data) VALUES (?, ?, ?)", (session_id, step, json.dumps(data)))
            conn.commit()
            metric_id = cursor.lastrowid
            row = conn.execute("SELECT * FROM metrics WHERE id = ?", (metric_id,)).fetchone()
            if row:
                d = dict(row)
                if d["data"]:
                    d["data"] = json.loads(d["data"])
                return d
        return None

    def append_episode(self, session_id: str, episode_data: dict[str, Any]):
        # Extract fields for dedicated columns, rest goes into 'data' JSON blob
        # The schema expects: id, session_id, step, task, is_correct, reward, data

        # We need to extract episode_id, task, etc from the dict
        # The 'episode_data' dict structure matches what UILogger sends:
        # { "episode_id": ..., "task": ..., "is_correct": ..., "reward": ..., "trajectories": ... }

        ep_id = episode_data.get("episode_id")
        step = episode_data.get("step")
        task = json.dumps(episode_data.get("task"))
        is_correct = episode_data.get("is_correct")
        reward = episode_data.get("reward")

        # 'data' column stores the full blob including trajectories
        full_data = json.dumps(episode_data)

        # Extract searchable text for full-text search
        search_text = extract_searchable_text(episode_data)

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO episodes (id, session_id, step, task, is_correct, reward, data, search_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ep_id, session_id, step, task, is_correct, reward, full_data, search_text),
            )
            conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row:
                d = dict(row)
                if d["config"]:
                    d["config"] = json.loads(d["config"])
                if d["source_metadata"]:
                    d["source_metadata"] = json.loads(d["source_metadata"])
                return d
        return None

    def get_all_sessions(self) -> list[dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
            results = []
            for row in rows:
                d = dict(row)
                # Parse JSON fields for convenience, or leave as string? Usually clean APIs return JSON objects.
                if d["config"]:
                    d["config"] = json.loads(d["config"])
                try:
                    if d.get("source_metadata"):
                        d["source_metadata"] = json.loads(d["source_metadata"])
                except (json.JSONDecodeError, TypeError):
                    pass
                results.append(d)
            return results

    def complete_session(self, session_id: str) -> dict[str, Any] | None:
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        with self._get_conn() as conn:
            # Check if session exists (optional, could just update)
            conn.execute("UPDATE sessions SET completed_at = ? WHERE id = ?", (now, session_id))
            conn.commit()
            # Fetch updated
            return self.get_session(session_id)

    def get_metrics(self, session_id: str) -> list[dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM metrics WHERE session_id = ? ORDER BY step", (session_id,)).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                if d["data"]:
                    d["data"] = json.loads(d["data"])
                results.append(d)
            return results

    def get_new_metrics(self, session_id: str, last_id: int) -> list[dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM metrics WHERE session_id = ? AND id > ? ORDER BY id", (session_id, last_id)).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                if d["data"]:
                    d["data"] = json.loads(d["data"])
                results.append(d)
            return results

    def get_episodes(self, session_id: str) -> list[dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM episodes WHERE session_id = ? ORDER BY step", (session_id,)).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                if d["data"]:
                    d["data"] = json.loads(d["data"])
                if d["task"]:
                    d["task"] = json.loads(d["task"])
                results.append(d)
            return results

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
            if row:
                d = dict(row)
                if d["data"]:
                    d["data"] = json.loads(d["data"])
                if d["task"]:
                    d["task"] = json.loads(d["task"])
                return d
        return None

    def search_episodes(self, query: str, session_id: str | None = None, limit: int = 50, step: int | None = None) -> dict[str, Any]:
        """Search episodes using LIKE (basic substring matching).

        Returns:
            Dict with 'episodes' list and 'matched_terms' (original query terms for SQLite)
        """
        with self._get_conn() as conn:
            sql = "SELECT * FROM episodes WHERE search_text LIKE ?"
            params: list = [f"%{query}%"]

            if session_id:
                sql += " AND session_id = ?"
                params.append(session_id)

            if step is not None:
                sql += " AND step = ?"
                params.append(step)

            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(sql, params).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                if d["data"]:
                    d["data"] = json.loads(d["data"])
                if d["task"]:
                    d["task"] = json.loads(d["task"])
                results.append(d)

            # For SQLite, return original query terms (no stemming)
            matched_terms = query.lower().split()

            return {
                "episodes": results,
                "matched_terms": matched_terms,
            }

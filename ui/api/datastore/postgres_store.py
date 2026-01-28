"""PostgreSQL implementation of DataStore."""

import json
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from .base import DataStore, extract_searchable_text


class PostgresStore(DataStore):
    """PostgreSQL-backed data store."""

    def __init__(self, url: str):
        self.url = url
        self._conn = None

    @contextmanager
    def _get_conn(self):
        """Get a database connection with RealDictCursor for dict-like row access."""
        conn = psycopg2.connect(self.url, cursor_factory=RealDictCursor)
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self):
        """Initialize the database schema."""
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                # Sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        project TEXT NOT NULL,
                        experiment TEXT NOT NULL,
                        config JSONB,
                        source_metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP
                    )
                """)

                # Metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT REFERENCES sessions(id),
                        step INTEGER,
                        data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Episodes table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS episodes (
                        id TEXT PRIMARY KEY,
                        session_id TEXT REFERENCES sessions(id),
                        step INTEGER,
                        task JSONB,
                        is_correct BOOLEAN,
                        reward REAL,
                        data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create indexes for better query performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_session_id ON metrics(session_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_episodes_session_id ON episodes(session_id)
                """)

                # Migration: Add search_text column for full-text search
                cursor.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'episodes' AND column_name = 'search_text'
                """)
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE episodes ADD COLUMN search_text TEXT")
                    # GIN index for fast full-text search
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_episodes_search
                        ON episodes USING GIN(to_tsvector('english', COALESCE(search_text, '')))
                    """)

                conn.commit()

    def reset(self):
        """Reset the data store by dropping and recreating all tables."""
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS episodes CASCADE")
                cursor.execute("DROP TABLE IF EXISTS metrics CASCADE")
                cursor.execute("DROP TABLE IF EXISTS sessions CASCADE")
                conn.commit()
        self.init_db()

    def create_session(self, project: str, experiment: str, config: dict[str, Any], source_metadata: dict[str, Any]) -> str:
        """Create a new training session."""
        session_id = str(uuid.uuid4())
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO sessions (id, project, experiment, config, source_metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (session_id, project, experiment, json.dumps(config), json.dumps(source_metadata)),
                )
                conn.commit()
        return session_id

    def log_metrics(self, session_id: str, step: int, data: dict[str, Any]) -> dict[str, Any] | None:
        """Log metrics for a session."""
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO metrics (session_id, step, data)
                    VALUES (%s, %s, %s)
                    RETURNING id, session_id, step, data, created_at
                    """,
                    (session_id, step, json.dumps(data)),
                )
                row = cursor.fetchone()
                conn.commit()
                if row:
                    return dict(row)
        return None

    def append_episode(self, session_id: str, episode_data: dict[str, Any]):
        """Append an episode to a session."""
        ep_id = episode_data.get("episode_id")
        step = episode_data.get("step")
        task = json.dumps(episode_data.get("task"))
        is_correct = episode_data.get("is_correct")
        reward = episode_data.get("reward")
        full_data = json.dumps(episode_data)

        # Extract searchable text for full-text search
        search_text = extract_searchable_text(episode_data)

        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO episodes (id, session_id, step, task, is_correct, reward, data, search_text)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (ep_id, session_id, step, task, is_correct, reward, full_data, search_text),
                )
                conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve session details."""
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
        return None

    def get_all_sessions(self) -> list[dict[str, Any]]:
        """Retrieve all sessions."""
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def complete_session(self, session_id: str) -> dict[str, Any] | None:
        """Mark a session as completed."""
        now = datetime.now(UTC).isoformat()
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE sessions SET completed_at = %s WHERE id = %s",
                    (now, session_id),
                )
                conn.commit()
        return self.get_session(session_id)

    def get_metrics(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve metrics for a session."""
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM metrics WHERE session_id = %s ORDER BY step",
                    (session_id,),
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def get_new_metrics(self, session_id: str, last_id: int) -> list[dict[str, Any]]:
        """Retrieve metrics since last_id."""
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM metrics WHERE session_id = %s AND id > %s ORDER BY id",
                    (session_id, last_id),
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def get_episodes(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve episodes for a session."""
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM episodes WHERE session_id = %s ORDER BY step",
                    (session_id,),
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        """Retrieve a specific episode."""
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM episodes WHERE id = %s", (episode_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
        return None

    def search_episodes(self, query: str, session_id: str | None = None, limit: int = 50, step: int | None = None) -> dict[str, Any]:
        """Search episodes using PostgreSQL full-text search with ranking.

        Features:
        - Stemming: "programming" matches "programs", "programmed"
        - Ranking: Results sorted by relevance (ts_rank)
        - Boolean queries: "python & machine" (AND), "python | java" (OR)

        Returns:
            Dict with 'episodes' list and 'matched_terms' (stemmed query terms)
        """
        with self._get_conn() as conn:
            with conn.cursor() as cursor:
                # First, get the stemmed lexemes from the query
                cursor.execute("SELECT plainto_tsquery('english', %s)::text AS query_text", (query,))
                query_result = cursor.fetchone()
                # Parse lexemes from tsquery string like "'subtract' & 'number'"
                query_text = query_result["query_text"] if query_result else ""
                # Extract terms between single quotes
                import re

                matched_terms = re.findall(r"'([^']+)'", query_text)

                # Build query with full-text search and ranking
                sql = """
                    SELECT *,
                           ts_rank(
                               to_tsvector('english', COALESCE(search_text, '')),
                               plainto_tsquery('english', %s)
                           ) AS rank
                    FROM episodes
                    WHERE to_tsvector('english', COALESCE(search_text, ''))
                          @@ plainto_tsquery('english', %s)
                """
                params: list = [query, query]

                if session_id:
                    sql += " AND session_id = %s"
                    params.append(session_id)

                if step is not None:
                    sql += " AND step = %s"
                    params.append(step)

                sql += " ORDER BY rank DESC LIMIT %s"
                params.append(limit)

                cursor.execute(sql, params)
                rows = cursor.fetchall()

                return {
                    "episodes": [dict(row) for row in rows],
                    "matched_terms": matched_terms,
                }

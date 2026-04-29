"""Tests for the local sqlite reader in :mod:`rllm.eval.trace_loader`."""

import json
import sqlite3
import time
from pathlib import Path

import pytest

from rllm.eval import trace_loader


def _seed_db(db_path: Path, rows: list[tuple[str, str, dict, float | None]]) -> None:
    """Create the gateway's sqlite schema and insert *rows*.

    Each row is ``(trace_id, session_id, data_dict, created_at_or_None)``.
    Mirrors the writes performed by ``rllm_model_gateway.store.sqlite_store``.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE traces (trace_id TEXT PRIMARY KEY, data TEXT NOT NULL, created_at REAL NOT NULL)")
        conn.execute("CREATE TABLE trace_sessions (trace_id TEXT NOT NULL, session_id TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY (trace_id, session_id))")
        for trace_id, session_id, data, ts in rows:
            now = ts if ts is not None else time.time()
            conn.execute(
                "INSERT INTO traces (trace_id, data, created_at) VALUES (?, ?, ?)",
                (trace_id, json.dumps(data), now),
            )
            conn.execute(
                "INSERT INTO trace_sessions (trace_id, session_id, created_at) VALUES (?, ?, ?)",
                (trace_id, session_id, now),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def populated_db(tmp_path):
    db = tmp_path / "traces.db"
    _seed_db(
        db,
        [
            ("t1", "eval-0", {"trace_id": "t1", "session_id": "eval-0", "model": "m"}, 100.0),
            ("t2", "eval-0", {"trace_id": "t2", "session_id": "eval-0", "model": "m"}, 101.0),
            ("t3", "eval-1", {"trace_id": "t3", "session_id": "eval-1", "model": "m"}, 102.0),
        ],
    )
    return db


class TestListSessionIds:
    def test_returns_distinct_ordered_by_first_seen(self, populated_db):
        assert trace_loader.list_session_ids(populated_db) == ["eval-0", "eval-1"]

    def test_missing_db_returns_empty(self, tmp_path):
        assert trace_loader.list_session_ids(tmp_path / "nope.db") == []


class TestSessionSummaries:
    def test_counts_and_timestamps(self, populated_db):
        summary = trace_loader.session_summaries(populated_db)
        assert summary["eval-0"] == {"trace_count": 2, "first_at": 100.0, "last_at": 101.0}
        assert summary["eval-1"] == {"trace_count": 1, "first_at": 102.0, "last_at": 102.0}

    def test_missing_db_returns_empty(self, tmp_path):
        assert trace_loader.session_summaries(tmp_path / "nope.db") == {}


class TestGetTraces:
    def test_basic_returns_decoded(self, populated_db):
        rows = trace_loader.get_traces(populated_db, "eval-0")
        assert [r["trace_id"] for r in rows] == ["t1", "t2"]
        # Each row carries the synthetic _created_at cursor.
        assert rows[0]["_created_at"] == 100.0
        assert rows[1]["_created_at"] == 101.0

    def test_since_is_strict(self, populated_db):
        # since is strict (>): passing 100.0 returns t2 only, not t1.
        rows = trace_loader.get_traces(populated_db, "eval-0", since=100.0)
        assert [r["trace_id"] for r in rows] == ["t2"]

    def test_limit(self, populated_db):
        rows = trace_loader.get_traces(populated_db, "eval-0", limit=1)
        assert [r["trace_id"] for r in rows] == ["t1"]

    def test_unknown_session_returns_empty(self, populated_db):
        assert trace_loader.get_traces(populated_db, "ghost") == []

    def test_missing_db_returns_empty(self, tmp_path):
        assert trace_loader.get_traces(tmp_path / "nope.db", "eval-0") == []


class TestCountTraces:
    def test_count(self, populated_db):
        assert trace_loader.count_traces(populated_db, "eval-0") == 2
        assert trace_loader.count_traces(populated_db, "eval-1") == 1
        assert trace_loader.count_traces(populated_db, "absent") == 0

    def test_missing_db_returns_zero(self, tmp_path):
        assert trace_loader.count_traces(tmp_path / "nope.db", "eval-0") == 0

"""Tests for the local sqlite reader in :mod:`rllm.eval.trace_loader`."""

import json
import sqlite3
import time
from pathlib import Path

import pytest

from rllm.eval import trace_loader


def _seed_db(
    db_path: Path,
    rows: list,
    *,
    include_run_id: bool = True,
    runs: list | None = None,
) -> None:
    """Create the gateway's sqlite schema and insert *rows*.

    With ``include_run_id=True`` (default) each row is
    ``(trace_id, session_id, run_id, data_dict, created_at_or_None)`` and
    the schema includes the ``run_id`` column plus the ``runs`` table.

    With ``include_run_id=False`` each row is
    ``(trace_id, session_id, data_dict, created_at_or_None)`` and the
    schema is the legacy form (no ``run_id`` column) — used to verify
    the loader's backwards-compat path.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE traces (trace_id TEXT PRIMARY KEY, data TEXT NOT NULL, created_at REAL NOT NULL)")
        if include_run_id:
            conn.execute(
                "CREATE TABLE trace_sessions (trace_id TEXT NOT NULL, session_id TEXT NOT NULL, run_id TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL, PRIMARY KEY (trace_id, session_id))"
            )
            conn.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, started_at REAL NOT NULL, ended_at REAL, metadata TEXT NOT NULL DEFAULT '{}')")
            for trace_id, session_id, run_id, data, ts in rows:
                now = ts if ts is not None else time.time()
                conn.execute(
                    "INSERT INTO traces (trace_id, data, created_at) VALUES (?, ?, ?)",
                    (trace_id, json.dumps(data), now),
                )
                conn.execute(
                    "INSERT INTO trace_sessions (trace_id, session_id, run_id, created_at) VALUES (?, ?, ?, ?)",
                    (trace_id, session_id, run_id, now),
                )
            for rid, started, ended, metadata in runs or []:
                conn.execute(
                    "INSERT INTO runs (run_id, started_at, ended_at, metadata) VALUES (?, ?, ?, ?)",
                    (rid, started, ended, json.dumps(metadata)),
                )
        else:
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
    """Single-run db: three traces in run-A, two sessions."""
    db = tmp_path / "traces.db"
    _seed_db(
        db,
        [
            ("t1", "eval-0", "run-A", {"trace_id": "t1", "session_id": "eval-0", "model": "m"}, 100.0),
            ("t2", "eval-0", "run-A", {"trace_id": "t2", "session_id": "eval-0", "model": "m"}, 101.0),
            ("t3", "eval-1", "run-A", {"trace_id": "t3", "session_id": "eval-1", "model": "m"}, 102.0),
        ],
        runs=[("run-A", 99.0, 110.0, {"benchmark": "gsm8k", "model": "m"})],
    )
    return db


@pytest.fixture
def two_run_db(tmp_path):
    """Cross-run db: same session_id (eval-0) used in two different runs."""
    db = tmp_path / "traces.db"
    _seed_db(
        db,
        [
            # run-A's eval-0
            ("a1", "eval-0", "run-A", {"trace_id": "a1"}, 100.0),
            ("a2", "eval-0", "run-A", {"trace_id": "a2"}, 101.0),
            # run-B's eval-0 — same session_id, different run
            ("b1", "eval-0", "run-B", {"trace_id": "b1"}, 200.0),
            ("b2", "eval-0", "run-B", {"trace_id": "b2"}, 201.0),
            ("b3", "eval-0", "run-B", {"trace_id": "b3"}, 202.0),
        ],
        runs=[
            ("run-A", 99.0, 110.0, {"benchmark": "gsm8k"}),
            ("run-B", 199.0, None, {"benchmark": "math500"}),
        ],
    )
    return db


@pytest.fixture
def legacy_db(tmp_path):
    """Db with the pre-run_id schema — exercises the loader's compat path."""
    db = tmp_path / "traces.db"
    # Old shape: 4-tuple (no run_id)
    _seed_db(
        db,
        [
            ("t1", "eval-0", {"trace_id": "t1"}, 100.0),
            ("t2", "eval-0", {"trace_id": "t2"}, 101.0),
        ],
        include_run_id=False,
    )
    return db


class TestListSessionIds:
    def test_returns_distinct_ordered_by_first_seen(self, populated_db):
        assert trace_loader.list_session_ids(populated_db) == ["eval-0", "eval-1"]

    def test_run_id_filter(self, two_run_db):
        # Both runs use session_id "eval-0"; filter narrows to one.
        assert trace_loader.list_session_ids(two_run_db, run_id="run-A") == ["eval-0"]
        assert trace_loader.list_session_ids(two_run_db, run_id="run-B") == ["eval-0"]
        # No filter: both rows collapsed (one distinct session_id).
        assert trace_loader.list_session_ids(two_run_db) == ["eval-0"]

    def test_legacy_schema_works(self, legacy_db):
        assert trace_loader.list_session_ids(legacy_db) == ["eval-0"]

    def test_missing_db_returns_empty(self, tmp_path):
        assert trace_loader.list_session_ids(tmp_path / "nope.db") == []


class TestSessionSummaries:
    def test_counts_and_timestamps(self, populated_db):
        summary = trace_loader.session_summaries(populated_db)
        assert summary["eval-0"] == {"trace_count": 2, "first_at": 100.0, "last_at": 101.0}
        assert summary["eval-1"] == {"trace_count": 1, "first_at": 102.0, "last_at": 102.0}

    def test_run_id_scopes_summary(self, two_run_db):
        a = trace_loader.session_summaries(two_run_db, run_id="run-A")
        assert a["eval-0"]["trace_count"] == 2
        b = trace_loader.session_summaries(two_run_db, run_id="run-B")
        assert b["eval-0"]["trace_count"] == 3

    def test_missing_db_returns_empty(self, tmp_path):
        assert trace_loader.session_summaries(tmp_path / "nope.db") == {}


class TestSessionSummariesByRun:
    def test_disambiguates_same_session_across_runs(self, two_run_db):
        rows = trace_loader.session_summaries_by_run(two_run_db)
        # Two rows: (eval-0, run-A) and (eval-0, run-B).
        keys = {(r["session_id"], r["run_id"]) for r in rows}
        assert keys == {("eval-0", "run-A"), ("eval-0", "run-B")}
        by_run = {r["run_id"]: r for r in rows}
        assert by_run["run-A"]["trace_count"] == 2
        assert by_run["run-B"]["trace_count"] == 3

    def test_legacy_schema_buckets_to_empty_run_id(self, legacy_db):
        rows = trace_loader.session_summaries_by_run(legacy_db)
        assert len(rows) == 1
        assert rows[0]["run_id"] == ""
        assert rows[0]["session_id"] == "eval-0"


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

    def test_run_id_scopes_traces(self, two_run_db):
        a = trace_loader.get_traces(two_run_db, "eval-0", run_id="run-A")
        assert [r["trace_id"] for r in a] == ["a1", "a2"]
        b = trace_loader.get_traces(two_run_db, "eval-0", run_id="run-B")
        assert [r["trace_id"] for r in b] == ["b1", "b2", "b3"]

    def test_no_run_id_is_cross_run(self, two_run_db):
        # Without a filter, both runs' rows for session_id eval-0 land together.
        rows = trace_loader.get_traces(two_run_db, "eval-0")
        assert [r["trace_id"] for r in rows] == ["a1", "a2", "b1", "b2", "b3"]

    def test_unknown_session_returns_empty(self, populated_db):
        assert trace_loader.get_traces(populated_db, "ghost") == []

    def test_missing_db_returns_empty(self, tmp_path):
        assert trace_loader.get_traces(tmp_path / "nope.db", "eval-0") == []


class TestCountTraces:
    def test_count(self, populated_db):
        assert trace_loader.count_traces(populated_db, "eval-0") == 2
        assert trace_loader.count_traces(populated_db, "eval-1") == 1
        assert trace_loader.count_traces(populated_db, "absent") == 0

    def test_run_id_scope(self, two_run_db):
        assert trace_loader.count_traces(two_run_db, "eval-0", run_id="run-A") == 2
        assert trace_loader.count_traces(two_run_db, "eval-0", run_id="run-B") == 3

    def test_missing_db_returns_zero(self, tmp_path):
        assert trace_loader.count_traces(tmp_path / "nope.db", "eval-0") == 0


class TestListRuns:
    def test_returns_metadata_ordered_by_started_desc(self, two_run_db):
        rows = trace_loader.list_runs(two_run_db)
        assert [r["run_id"] for r in rows] == ["run-B", "run-A"]
        a = next(r for r in rows if r["run_id"] == "run-A")
        assert a["metadata"] == {"benchmark": "gsm8k"}
        assert a["ended_at"] == 110.0
        assert a["started_at"] == 99.0
        b = next(r for r in rows if r["run_id"] == "run-B")
        assert b["ended_at"] is None  # still running

    def test_legacy_schema_returns_empty(self, legacy_db):
        # No `runs` table exists in the legacy schema.
        assert trace_loader.list_runs(legacy_db) == []

    def test_missing_db_returns_empty(self, tmp_path):
        assert trace_loader.list_runs(tmp_path / "nope.db") == []

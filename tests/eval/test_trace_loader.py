"""Tests for the local sqlite reader in :mod:`rllm.eval.trace_loader`.

The reader is a stdlib-only ``sqlite3`` consumer of the v2 gateway
schema (see ``rllm_model_gateway.store.sqlite_store``). Tests seed the
db directly so we don't drag in ``aiosqlite`` here.
"""

import json
import sqlite3
import time
from pathlib import Path

import pytest

from rllm.eval import trace_loader

_SCHEMA_V2_TRACES = """
CREATE TABLE traces (
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

_SCHEMA_V2_TRACE_SESSIONS = """
CREATE TABLE trace_sessions (
    trace_id   TEXT NOT NULL,
    session_id TEXT NOT NULL,
    run_id     TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    PRIMARY KEY (trace_id, session_id)
)
"""

_SCHEMA_V2_RUNS = """
CREATE TABLE runs (
    run_id     TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    ended_at   REAL,
    metadata   TEXT NOT NULL DEFAULT '{}'
)
"""


def _seed_db(
    db_path: Path,
    rows: list,
    *,
    runs: list | None = None,
) -> None:
    """Seed a v2-shaped db with traces.

    Each row: ``(trace_id, session_id, run_id, data_dict, created_at)``.
    Denormalized columns (``model``, ``harness``, ``latency_ms``,
    ``has_error``, ``step_id``) come from ``data_dict`` keys with
    sensible defaults so most tests can omit them.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA user_version = 2")
        conn.execute(_SCHEMA_V2_TRACES)
        conn.execute(_SCHEMA_V2_TRACE_SESSIONS)
        conn.execute(_SCHEMA_V2_RUNS)
        for trace_id, session_id, run_id, data, ts in rows:
            now = ts if ts is not None else time.time()
            model = data.get("model", "")
            harness = data.get("harness")
            latency_ms = int(data.get("latency_ms") or 0)
            has_error = int(data.get("has_error") or 0)
            step_id = data.get("step_id")
            conn.execute(
                """INSERT INTO traces
                   (trace_id, data, created_at, session_id, run_id,
                    model, harness, latency_ms, has_error, step_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (trace_id, json.dumps(data), now, session_id, run_id, model, harness, latency_ms, has_error, step_id),
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
def filterable_db(tmp_path):
    """Mixed models, harnesses, latencies — for filter-pushdown tests."""
    db = tmp_path / "traces.db"
    _seed_db(
        db,
        [
            ("t1", "s", "r", {"trace_id": "t1", "model": "gpt-5", "harness": "bash", "latency_ms": 50}, 100.0),
            ("t2", "s", "r", {"trace_id": "t2", "model": "claude", "harness": "bash", "latency_ms": 500}, 101.0),
            ("t3", "s", "r", {"trace_id": "t3", "model": "gpt-5", "harness": "claude-code", "latency_ms": 1500, "has_error": 1}, 102.0),
        ],
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
        rows = trace_loader.get_traces(two_run_db, "eval-0")
        assert [r["trace_id"] for r in rows] == ["a1", "a2", "b1", "b2", "b3"]

    def test_unknown_session_returns_empty(self, populated_db):
        assert trace_loader.get_traces(populated_db, "ghost") == []

    def test_missing_db_returns_empty(self, tmp_path):
        assert trace_loader.get_traces(tmp_path / "nope.db", "eval-0") == []


class TestQueryTraces:
    def test_filter_by_model(self, filterable_db):
        rows = trace_loader.query_traces(filterable_db, model="claude")
        assert [r["trace_id"] for r in rows] == ["t2"]

    def test_filter_by_harness(self, filterable_db):
        rows = trace_loader.query_traces(filterable_db, harness="bash")
        assert {r["trace_id"] for r in rows} == {"t1", "t2"}

    def test_filter_by_has_error(self, filterable_db):
        ok = trace_loader.query_traces(filterable_db, has_error=False)
        err = trace_loader.query_traces(filterable_db, has_error=True)
        assert {r["trace_id"] for r in ok} == {"t1", "t2"}
        assert {r["trace_id"] for r in err} == {"t3"}

    def test_filter_by_latency_range(self, filterable_db):
        rows = trace_loader.query_traces(filterable_db, latency_min=100, latency_max=1000)
        assert [r["trace_id"] for r in rows] == ["t2"]

    def test_combined_filters(self, filterable_db):
        rows = trace_loader.query_traces(filterable_db, model="gpt-5", harness="bash")
        assert [r["trace_id"] for r in rows] == ["t1"]

    def test_order_desc_default(self, filterable_db):
        rows = trace_loader.query_traces(filterable_db)
        assert [r["trace_id"] for r in rows] == ["t3", "t2", "t1"]

    def test_order_asc(self, filterable_db):
        rows = trace_loader.query_traces(filterable_db, order="ASC")
        assert [r["trace_id"] for r in rows] == ["t1", "t2", "t3"]

    def test_pagination_until_cursor(self, filterable_db):
        # Newest-first feed: t3 (102), t2 (101), t1 (100).
        page1 = trace_loader.query_traces(filterable_db, limit=2)
        assert [r["trace_id"] for r in page1] == ["t3", "t2"]
        # Next page using until = oldest seen.
        cursor = page1[-1]["_created_at"]
        page2 = trace_loader.query_traces(filterable_db, limit=2, until=cursor)
        assert [r["trace_id"] for r in page2] == ["t1"]

    def test_live_tail_since_cursor(self, filterable_db):
        # Strictly-after cursor → 102.0 returns nothing (only t3 has 102.0).
        rows = trace_loader.query_traces(filterable_db, since=102.0)
        assert rows == []
        # 101.0 returns the newer t3.
        rows = trace_loader.query_traces(filterable_db, since=101.0)
        assert [r["trace_id"] for r in rows] == ["t3"]

    def test_invalid_order(self, filterable_db):
        with pytest.raises(ValueError):
            trace_loader.query_traces(filterable_db, order="bogus")

    def test_missing_db_returns_empty(self, tmp_path):
        assert trace_loader.query_traces(tmp_path / "nope.db") == []


class TestCountTraces:
    def test_count_by_session(self, populated_db):
        assert trace_loader.count_traces(populated_db, "eval-0") == 2
        assert trace_loader.count_traces(populated_db, "eval-1") == 1
        assert trace_loader.count_traces(populated_db, "absent") == 0

    def test_count_with_filters(self, filterable_db):
        assert trace_loader.count_traces(filterable_db) == 3
        assert trace_loader.count_traces(filterable_db, model="gpt-5") == 2
        assert trace_loader.count_traces(filterable_db, has_error=True) == 1

    def test_run_id_scope(self, two_run_db):
        assert trace_loader.count_traces(two_run_db, "eval-0", run_id="run-A") == 2
        assert trace_loader.count_traces(two_run_db, "eval-0", run_id="run-B") == 3

    def test_missing_db_returns_zero(self, tmp_path):
        assert trace_loader.count_traces(tmp_path / "nope.db", "eval-0") == 0


class TestListFacets:
    def test_facets(self, filterable_db):
        f = trace_loader.list_facets(filterable_db)
        assert f["models"] == ["claude", "gpt-5"]
        assert f["harnesses"] == ["bash", "claude-code"]
        assert f["runs"] == ["r"]

    def test_missing_db(self, tmp_path):
        assert trace_loader.list_facets(tmp_path / "nope.db") == {"models": [], "harnesses": [], "runs": []}


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

    def test_missing_db_returns_empty(self, tmp_path):
        assert trace_loader.list_runs(tmp_path / "nope.db") == []

"""Smoke tests for the Sessions panel — global ``/traces`` + ``/facets`` endpoints.

The endpoints proxy ``rllm.eval.trace_loader`` against the gateway
sqlite store. We point ``RLLM_GATEWAY_DB`` at a temporary v2-shaped
file so tests don't rely on the gateway service being running.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rllm.console import mount_console


def _seed_db(path: Path) -> None:
    """Seed a v2-shaped gateway db. Mirrors ``rllm_model_gateway.store.sqlite_store``."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        PRAGMA user_version = 2;
        CREATE TABLE traces (
            trace_id   TEXT PRIMARY KEY,
            data       TEXT NOT NULL,
            created_at REAL NOT NULL,
            session_id TEXT NOT NULL DEFAULT '',
            run_id     TEXT NOT NULL DEFAULT '',
            model      TEXT NOT NULL DEFAULT '',
            harness    TEXT,
            latency_ms INTEGER NOT NULL DEFAULT 0,
            has_error  INTEGER NOT NULL DEFAULT 0,
            step_id    INTEGER
        );
        CREATE TABLE trace_sessions (
            trace_id   TEXT NOT NULL,
            session_id TEXT NOT NULL,
            run_id     TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            PRIMARY KEY (trace_id, session_id)
        );
        CREATE TABLE runs (
            run_id     TEXT PRIMARY KEY,
            started_at REAL,
            ended_at   REAL,
            metadata   TEXT
        );
        """
    )
    now = time.time()
    rows = [
        # (trace_id, session_id, run_id, model, harness, latency_ms, has_error, ts_offset)
        ("t1", "sess-1", "run-A", "gpt-5", "bash", 50, 0, 0),
        ("t2", "sess-1", "run-A", "gpt-5", "bash", 100, 0, 1),
        ("t3", "sess-1", "run-A", "claude", "bash", 1500, 0, 2),
        ("t4", "sess-2", "run-A", "gpt-5", "claude-code", 200, 1, 3),
        ("t5", "sess-3", "run-B", "claude", "bash", 800, 0, 4),
    ]
    for tid, sid, rid, model, harness, lat, err, off in rows:
        ts = now - 100 + off
        conn.execute(
            """INSERT INTO traces
               (trace_id, data, created_at, session_id, run_id, model, harness, latency_ms, has_error, step_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tid,
                json.dumps({"trace_id": tid, "model": model, "latency_ms": lat}),
                ts,
                sid,
                rid,
                model,
                harness,
                lat,
                err,
                None,
            ),
        )
        conn.execute(
            "INSERT INTO trace_sessions VALUES (?, ?, ?, ?)",
            (tid, sid, rid, ts),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    db = tmp_path / "traces.db"
    _seed_db(db)
    monkeypatch.setenv("RLLM_GATEWAY_DB", str(db))

    app = FastAPI()
    mount_console(app, eval_results_root=tmp_path)
    return TestClient(app)


def test_traces_default_returns_all_newest_first(client: TestClient) -> None:
    r = client.get("/console/api/panels/sessions/traces")
    assert r.status_code == 200
    rows = r.json()
    assert [t["trace_id"] for t in rows] == ["t5", "t4", "t3", "t2", "t1"]
    for t in rows:
        assert "_created_at" in t


def test_filter_by_run_id(client: TestClient) -> None:
    rows = client.get("/console/api/panels/sessions/traces?run_id=run-B").json()
    assert {t["trace_id"] for t in rows} == {"t5"}


def test_filter_by_session_id(client: TestClient) -> None:
    rows = client.get("/console/api/panels/sessions/traces?session_id=sess-1").json()
    assert {t["trace_id"] for t in rows} == {"t1", "t2", "t3"}


def test_filter_by_model(client: TestClient) -> None:
    rows = client.get("/console/api/panels/sessions/traces?model=claude").json()
    assert {t["trace_id"] for t in rows} == {"t3", "t5"}


def test_filter_by_harness(client: TestClient) -> None:
    rows = client.get("/console/api/panels/sessions/traces?harness=claude-code").json()
    assert {t["trace_id"] for t in rows} == {"t4"}


def test_filter_by_has_error(client: TestClient) -> None:
    err = client.get("/console/api/panels/sessions/traces?has_error=true").json()
    ok = client.get("/console/api/panels/sessions/traces?has_error=false").json()
    assert {t["trace_id"] for t in err} == {"t4"}
    assert {t["trace_id"] for t in ok} == {"t1", "t2", "t3", "t5"}


def test_filter_by_latency_range(client: TestClient) -> None:
    rows = client.get(
        "/console/api/panels/sessions/traces?latency_min=100&latency_max=1000",
    ).json()
    assert {t["trace_id"] for t in rows} == {"t2", "t4", "t5"}


def test_combined_filters(client: TestClient) -> None:
    rows = client.get(
        "/console/api/panels/sessions/traces?run_id=run-A&model=gpt-5",
    ).json()
    assert {t["trace_id"] for t in rows} == {"t1", "t2", "t4"}


def test_pagination_until_cursor(client: TestClient) -> None:
    page1 = client.get("/console/api/panels/sessions/traces?limit=2").json()
    assert [t["trace_id"] for t in page1] == ["t5", "t4"]
    cursor = page1[-1]["_created_at"]
    page2 = client.get(f"/console/api/panels/sessions/traces?limit=2&until={cursor}").json()
    assert [t["trace_id"] for t in page2] == ["t3", "t2"]


def test_live_tail_since_cursor(client: TestClient) -> None:
    all_rows = client.get("/console/api/panels/sessions/traces?order=ASC").json()
    cursor = all_rows[-1]["_created_at"]
    later = client.get(f"/console/api/panels/sessions/traces?since={cursor}&order=ASC").json()
    assert later == []


def test_facets(client: TestClient) -> None:
    f = client.get("/console/api/panels/sessions/facets").json()
    assert f["models"] == ["claude", "gpt-5"]
    assert f["harnesses"] == ["bash", "claude-code"]
    assert f["runs"] == ["run-A", "run-B"]


def test_legacy_endpoints_are_gone(client: TestClient) -> None:
    """``/runs`` and ``/sessions`` were hard-deleted in Phase 2."""
    assert client.get("/console/api/panels/sessions/runs").status_code == 404
    assert client.get("/console/api/panels/sessions/sessions").status_code == 404

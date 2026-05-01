"""Smoke tests for the Sessions panel — runs/sessions/traces endpoints.

The endpoints proxy ``rllm.eval.trace_loader`` against the gateway
sqlite store. We point ``RLLM_GATEWAY_DB`` at a temporary file with a
hand-rolled minimal schema so tests don't rely on the gateway service
being installed/initialised.
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
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE traces (
            trace_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE trace_sessions (
            trace_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            run_id TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            PRIMARY KEY (trace_id, session_id)
        );
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            started_at REAL,
            ended_at REAL,
            metadata TEXT
        );
        """
    )
    now = time.time()
    conn.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?)",
        ("run-A", now - 100, None, json.dumps({"agent": "x"})),
    )
    for i, sid in enumerate(["sess-1", "sess-2"]):
        for j in range(3):
            tid = f"t-{sid}-{j}"
            conn.execute(
                "INSERT INTO traces VALUES (?, ?, ?)",
                (tid, json.dumps({"model": "m", "i": j}), now - 100 + j),
            )
            conn.execute(
                "INSERT INTO trace_sessions VALUES (?, ?, ?, ?)",
                (tid, sid, "run-A", now - 100 + j + i * 10),
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


def test_runs_endpoint(client: TestClient) -> None:
    r = client.get("/console/api/panels/sessions/runs")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "run-A"
    assert row["session_count"] == 2
    assert row["trace_count"] == 6  # 2 sessions * 3 traces


def test_sessions_endpoint_per_run(client: TestClient) -> None:
    r = client.get("/console/api/panels/sessions/sessions?run_id=run-A")
    assert r.status_code == 200
    rows = r.json()
    assert {x["session_id"] for x in rows} == {"sess-1", "sess-2"}
    assert all(x["run_id"] == "run-A" for x in rows)
    assert all(x["trace_count"] == 3 for x in rows)


def test_sessions_endpoint_cross_run(client: TestClient) -> None:
    """Without ``run_id``, returns every (run_id, session_id) pair."""
    r = client.get("/console/api/panels/sessions/sessions")
    assert r.status_code == 200
    rows = r.json()
    assert {(x["run_id"], x["session_id"]) for x in rows} == {
        ("run-A", "sess-1"),
        ("run-A", "sess-2"),
    }


def test_traces_endpoint(client: TestClient) -> None:
    r = client.get(
        "/console/api/panels/sessions/traces?session_id=sess-1&run_id=run-A",
    )
    assert r.status_code == 200
    traces = r.json()
    assert len(traces) == 3
    # Each trace is the JSON we stored, plus a synthetic _created_at.
    for t in traces:
        assert "_created_at" in t
        assert t["model"] == "m"


def test_traces_requires_session_id(client: TestClient) -> None:
    r = client.get("/console/api/panels/sessions/traces")
    # FastAPI returns 422 for missing required query params.
    assert r.status_code == 422


def test_traces_since_filter(client: TestClient) -> None:
    """``since`` only returns traces newer than the cursor."""
    # First read all to find a midpoint cursor.
    all_traces = client.get(
        "/console/api/panels/sessions/traces?session_id=sess-1&run_id=run-A",
    ).json()
    cursor = all_traces[1]["_created_at"]
    r = client.get(
        f"/console/api/panels/sessions/traces?session_id=sess-1&run_id=run-A&since={cursor}",
    )
    assert r.status_code == 200
    later = r.json()
    assert len(later) == 1  # only the third trace, strictly after cursor

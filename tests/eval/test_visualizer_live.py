"""Tests for the live-mode endpoints in :mod:`rllm.eval.visualizer`.

Covers ``GET /api/runs/{id}/traces`` (delta-fetch with ``since`` cursor)
and ``GET /api/runs/{id}/live`` (in-flight derivation from
``tasks.jsonl`` minus already-written ``episodes/``).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import httpx
import pytest

from rllm.eval import visualizer


def _seed_traces_db(db_path: Path, rows: list[tuple[str, str, dict, float]]) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE traces (trace_id TEXT PRIMARY KEY, data TEXT NOT NULL, created_at REAL NOT NULL)")
        conn.execute("CREATE TABLE trace_sessions (trace_id TEXT NOT NULL, session_id TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY (trace_id, session_id))")
        for trace_id, session_id, data, ts in rows:
            conn.execute(
                "INSERT INTO traces (trace_id, data, created_at) VALUES (?, ?, ?)",
                (trace_id, json.dumps(data), ts),
            )
            conn.execute(
                "INSERT INTO trace_sessions (trace_id, session_id, created_at) VALUES (?, ?, ?)",
                (trace_id, session_id, ts),
            )
        conn.commit()
    finally:
        conn.close()


def _write_episode(episodes_dir: Path, idx: int, task_id: str) -> None:
    episodes_dir.mkdir(parents=True, exist_ok=True)
    path = episodes_dir / f"episode_{idx:06d}_{task_id}.json"
    path.write_text(json.dumps({"eval_idx": idx, "task": {"id": task_id}, "trajectories": []}))


def _write_tasks_jsonl(run_dir: Path, lines: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "tasks.jsonl", "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


@pytest.fixture
def root_with_run(tmp_path):
    """Create ``<root>/<run_id>/`` populated with tasks.jsonl, episodes/, traces.db."""
    root = tmp_path / "eval_results"
    run_id = "demo-run"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)

    # tasks.jsonl: 3 tasks started.
    now = time.time()
    _write_tasks_jsonl(
        run_dir,
        [
            {"idx": 0, "session_id": "eval-0", "task_id": "t-zero", "instruction": "Solve A", "started_at": now - 10.0},
            {"idx": 1, "session_id": "eval-1", "task_id": "t-one", "instruction": "Solve B", "started_at": now - 5.0},
            {"idx": 2, "session_id": "eval-2", "task_id": "t-two", "instruction": "Solve C", "started_at": now - 1.0},
        ],
    )

    # Only idx=0 has a written episode → idx=1, 2 are in-flight.
    _write_episode(run_dir / "episodes", 0, "t-zero")

    # Traces: 2 for eval-0 (finished), 3 for eval-1 (in-flight), 0 for eval-2.
    _seed_traces_db(
        run_dir / "traces.db",
        [
            ("a1", "eval-0", {"trace_id": "a1", "session_id": "eval-0"}, 1000.0),
            ("a2", "eval-0", {"trace_id": "a2", "session_id": "eval-0"}, 1001.0),
            ("b1", "eval-1", {"trace_id": "b1", "session_id": "eval-1"}, 1002.0),
            ("b2", "eval-1", {"trace_id": "b2", "session_id": "eval-1"}, 1003.0),
            ("b3", "eval-1", {"trace_id": "b3", "session_id": "eval-1"}, 1004.0),
        ],
    )

    return root, run_id


@pytest.fixture
def server(root_with_run):
    root, _ = root_with_run

    def html_factory():
        return "<html></html>"

    handler_cls = visualizer._make_handler(root, html_factory)
    srv = visualizer._ThreadingServer(("127.0.0.1", 0), handler_cls)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        srv.server_close()


class TestTracesEndpoint:
    def test_returns_traces_for_session(self, server, root_with_run):
        _, run_id = root_with_run
        resp = httpx.get(f"{server}/api/runs/{run_id}/traces", params={"session_id": "eval-1"})
        assert resp.status_code == 200
        rows = resp.json()
        assert [r["trace_id"] for r in rows] == ["b1", "b2", "b3"]
        assert rows[0]["_created_at"] == 1002.0

    def test_since_filters_strictly(self, server, root_with_run):
        _, run_id = root_with_run
        resp = httpx.get(
            f"{server}/api/runs/{run_id}/traces",
            params={"session_id": "eval-1", "since": "1003.0"},
        )
        assert resp.status_code == 200
        assert [r["trace_id"] for r in resp.json()] == ["b3"]

    def test_session_id_required(self, server, root_with_run):
        _, run_id = root_with_run
        resp = httpx.get(f"{server}/api/runs/{run_id}/traces")
        assert resp.status_code == 400

    def test_unsafe_run_id_rejected(self, server):
        resp = httpx.get(f"{server}/api/runs/..%2Fetc/traces", params={"session_id": "x"})
        assert resp.status_code == 400


class TestLiveEndpoint:
    def test_in_flight_derivation(self, server, root_with_run):
        _, run_id = root_with_run
        resp = httpx.get(f"{server}/api/runs/{run_id}/live")
        assert resp.status_code == 200
        payload = resp.json()
        # idx=0 finished (episode written) → not in_flight
        # idx=1, 2 still in_flight
        idxs = [r["idx"] for r in payload["in_flight"]]
        assert idxs == [1, 2]
        assert payload["finished_count"] == 1
        assert payload["started_count"] == 3

    def test_in_flight_carries_trace_count(self, server, root_with_run):
        _, run_id = root_with_run
        payload = httpx.get(f"{server}/api/runs/{run_id}/live").json()
        by_idx = {r["idx"]: r for r in payload["in_flight"]}
        assert by_idx[1]["trace_count"] == 3
        assert by_idx[2]["trace_count"] == 0
        assert by_idx[1]["instruction"] == "Solve B"
        assert by_idx[1]["session_id"] == "eval-1"

    def test_unknown_run_returns_404(self, server):
        resp = httpx.get(f"{server}/api/runs/no-such-run/live")
        assert resp.status_code == 404

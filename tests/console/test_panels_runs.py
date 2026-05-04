"""Smoke tests + path-safety for the Runs panel endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rllm.console import mount_console


def _mk_run(root: Path, run_id: str, *, episode_idx: int = 0, with_results: bool = True) -> Path:
    run_dir = root / run_id
    episodes = run_dir / "episodes"
    episodes.mkdir(parents=True)

    if with_results:
        (run_dir / "results.json").write_text(
            json.dumps(
                {
                    "dataset_name": "test-bench",
                    "model": "test-model",
                    "agent": "test-agent",
                    "score": 0.75,
                    "correct": 3,
                    "total": 4,
                    "errors": 0,
                }
            ),
            encoding="utf-8",
        )

    (episodes / f"episode_{episode_idx:06d}_test.json").write_text(
        json.dumps(
            {
                "id": "ep-0",
                "eval_idx": episode_idx,
                "is_correct": True,
                "termination_reason": "done",
                "task": {"id": "task-0", "instruction": "do thing"},
                "trajectories": [{"reward": 0.9, "steps": [{"input": "x", "output": "y"}]}],
            }
        ),
        encoding="utf-8",
    )
    return run_dir


@pytest.fixture
def client(tmp_path: Path) -> tuple[TestClient, Path]:
    eval_root = tmp_path / "eval_results"
    eval_root.mkdir()
    _mk_run(eval_root, "bench_model_20260101_120000")

    app = FastAPI()
    mount_console(app, eval_results_root=eval_root)
    return TestClient(app), eval_root


def test_list_runs_returns_summaries(client: tuple[TestClient, Path]) -> None:
    c, _ = client
    r = c.get("/console/api/panels/runs")
    assert r.status_code == 200
    runs = r.json()
    assert len(runs) == 1
    row = runs[0]
    assert row["id"] == "bench_model_20260101_120000"
    assert row["benchmark"] == "test-bench"
    assert row["score"] == 0.75
    assert row["status"] == "completed"
    assert row["n_episodes"] == 1


def test_episode_index(client: tuple[TestClient, Path]) -> None:
    c, _ = client
    r = c.get("/console/api/panels/runs/bench_model_20260101_120000/index")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["task_id"] == "task-0"
    assert rows[0]["is_correct"] is True
    assert rows[0]["n_steps"] == 1


def test_episode_file(client: tuple[TestClient, Path]) -> None:
    c, _ = client
    r = c.get(
        "/console/api/panels/runs/bench_model_20260101_120000/episodes/episode_000000_test.json",
    )
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "ep-0"


def test_live_payload_structure(client: tuple[TestClient, Path]) -> None:
    c, _ = client
    r = c.get("/console/api/panels/runs/bench_model_20260101_120000/live")
    assert r.status_code == 200
    body = r.json()
    # Liveness now sources started/ended from the gateway runs table;
    # ``in_flight`` is derived from ``ended_at IS NULL``. Sessions list
    # comes from the trace store. Tasks.jsonl backs ``in_flight_tasks``.
    assert set(body) == {
        "run_id",
        "started_at",
        "ended_at",
        "in_flight",
        "sessions",
        "in_flight_tasks",
        "finished_count",
        "started_count",
    }
    # No gateway db in this test → in_flight is False and sessions empty.
    assert body["in_flight"] is False
    assert body["sessions"] == []


def test_path_traversal_rejected(client: tuple[TestClient, Path]) -> None:
    """Run ids with .., /, \\, and chars outside [A-Za-z0-9._-] are rejected."""
    c, _ = client
    # Routes that don't even match (multi-segment) → 404. Route that
    # matches but fails validation → 400.
    assert c.get("/console/api/panels/runs/bad:run/index").status_code == 400
    assert c.get("/console/api/panels/runs/bad run/index").status_code == 400
    # Multi-segment via URL-decoded slash doesn't match the single
    # path-param route at all.
    assert c.get("/console/api/panels/runs/a%2Fb/index").status_code == 404


def test_episode_filename_validated(client: tuple[TestClient, Path]) -> None:
    c, _ = client
    # Filenames must match `episode_*.json` regex.
    r = c.get(
        "/console/api/panels/runs/bench_model_20260101_120000/episodes/results.json",
    )
    assert r.status_code == 400


def test_unknown_run_404(client: tuple[TestClient, Path]) -> None:
    c, _ = client
    assert c.get("/console/api/panels/runs/no-such-run/index").status_code == 404


def test_run_without_aggregate_still_listed(tmp_path: Path) -> None:
    eval_root = tmp_path / "eval_results"
    eval_root.mkdir()
    _mk_run(eval_root, "incomplete_run_20260101_000000", with_results=False)

    app = FastAPI()
    mount_console(app, eval_results_root=eval_root)
    runs = TestClient(app).get("/console/api/panels/runs").json()
    assert len(runs) == 1
    assert runs[0]["status"] == "incomplete"
    assert runs[0]["score"] is None

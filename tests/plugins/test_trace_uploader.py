"""U.1-U.4 — unit tests for rllm_trace_uploader."""

from __future__ import annotations

import json
import os
import sqlite3
import time

import pytest


def test_schema_parse_row() -> None:
    """U.1: TraceRow.from_sqlite parses stored data JSON blob."""
    from rllm_trace_uploader.schema import TraceRow

    data = {
        "trace_id": "abc",
        "session_id": "sess-1",
        "model": "qwen3.5-9b",
        "messages": [{"role": "user", "content": "hi"}],
        "response_message": {"role": "assistant", "content": "hello"},
        "latency_ms": 123.4,
        "token_counts": {"prompt": 3, "completion": 2},
        "finish_reason": "stop",
        "metadata": {"n_turns": 1},
        "raw_request": {"model": "qwen"},
        "raw_response": {"choices": [{}]},
    }
    row = TraceRow.from_sqlite(
        rowid=42,
        trace_id="abc",
        session_id="sess-1",
        data_json=json.dumps(data),
        created_at=1000.0,
    )
    assert row.trace_id == "abc"
    assert row.session_id == "sess-1"
    assert row.model == "qwen3.5-9b"
    assert row.messages == [{"role": "user", "content": "hi"}]
    assert row.response_message["content"] == "hello"
    assert row.latency_ms == pytest.approx(123.4)
    assert row.token_counts == {"prompt": 3, "completion": 2}
    assert row.finish_reason == "stop"
    assert row.metadata["n_turns"] == 1
    assert row.raw_request == {"model": "qwen"}
    assert row.rowid == 42


def test_schema_parse_missing_fields() -> None:
    """TraceRow tolerates partial data (older schema versions)."""
    from rllm_trace_uploader.schema import TraceRow

    row = TraceRow.from_sqlite(
        rowid=1,
        trace_id="t",
        session_id="s",
        data_json="{}",
        created_at=0.0,
    )
    assert row.messages == []
    assert row.response_message == {}
    assert row.latency_ms == 0.0
    assert row.token_counts == {}
    assert row.finish_reason is None


def test_sidecar_reader_mtime_cache(tmp_path) -> None:
    """U.2: SidecarReader mtime-cache invalidates when file changes."""
    from rllm_trace_uploader.sidecar import SidecarReader

    root = tmp_path
    step_path = root / "current_step.txt"
    run_path = root / "wandb_run_id.txt"
    step_path.write_text("5")
    run_path.write_text("me/proj/run123")

    reader = SidecarReader(str(root))
    assert reader.get_current_step() == 5
    assert reader.get_run_id() == "me/proj/run123"

    # Cached read (no file change) — same value
    assert reader.get_current_step() == 5

    # Update file & mtime, verify re-read
    time.sleep(0.02)  # ensure mtime changes
    step_path.write_text("42")
    os.utime(step_path, None)
    assert reader.get_current_step() == 42


def test_sidecar_reader_missing_files(tmp_path) -> None:
    """SidecarReader returns None gracefully when files don't exist."""
    from rllm_trace_uploader.sidecar import SidecarReader

    reader = SidecarReader(str(tmp_path))
    assert reader.get_current_step() is None
    assert reader.get_run_id() is None
    assert reader.get_latest_checkpoint() is None


def test_sidecar_reader_checkpoint_jsonl_last(tmp_path) -> None:
    """SidecarReader.get_latest_checkpoint returns the tail line."""
    from rllm_trace_uploader.sidecar import SidecarReader

    p = tmp_path / "checkpoint_versions.jsonl"
    p.write_text(
        json.dumps({"step": 1, "ts": 100}) + "\n"
        + json.dumps({"step": 2, "ts": 200}) + "\n"
        + json.dumps({"step": 3, "ts": 300}) + "\n"
    )
    reader = SidecarReader(str(tmp_path))
    ckpt = reader.get_latest_checkpoint()
    assert ckpt is not None
    assert ckpt["step"] == 3


def test_state_atomic_write(tmp_path) -> None:
    """U.3: state file uses .tmp + rename for atomicity."""
    from rllm_trace_uploader.sidecar import SidecarReader
    from rllm_trace_uploader.uploader import TraceUploader

    db_path = str(tmp_path / "unused.db")
    state_path = str(tmp_path / "state.txt")
    sidecar = SidecarReader(str(tmp_path))
    up = TraceUploader(
        db_path=db_path,
        state_path=state_path,
        weave_project="test",
        sidecar=sidecar,
    )
    up._last_rowid = 7
    up.save_state()
    assert not os.path.exists(state_path + ".tmp")  # tmp removed
    with open(state_path) as f:
        assert f.read().strip() == "7"


def _make_fake_sqlite_db(db_path: str, n_rows: int) -> None:
    """Create a minimal `traces` + `trace_sessions` DB matching gateway schema."""
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE traces (trace_id TEXT PRIMARY KEY, data TEXT, created_at REAL)")
    conn.execute(
        "CREATE TABLE trace_sessions (trace_id TEXT, session_id TEXT, created_at REAL, PRIMARY KEY (trace_id, session_id))"
    )
    for i in range(n_rows):
        tid = f"trace-{i:04d}"
        sid = f"session-{i // 5}"
        data = json.dumps(
            {
                "trace_id": tid,
                "session_id": sid,
                "model": "qwen",
                "messages": [{"role": "user", "content": f"q{i}"}],
                "response_message": {"role": "assistant", "content": f"a{i}"},
                "latency_ms": 1.0 * i,
                "token_counts": {"prompt": 3, "completion": 2},
                "finish_reason": "stop",
                "metadata": {},
            }
        )
        conn.execute(
            "INSERT INTO traces (trace_id, data, created_at) VALUES (?, ?, ?)",
            (tid, data, 1000.0 + i),
        )
        conn.execute(
            "INSERT INTO trace_sessions (trace_id, session_id, created_at) VALUES (?, ?, ?)",
            (tid, sid, 1000.0 + i),
        )
    conn.commit()
    conn.close()


def test_fetch_new_traces_pagination(tmp_path) -> None:
    """U.4: fetch_new_traces respects limit and advances last_rowid."""
    import asyncio

    from rllm_trace_uploader.sidecar import SidecarReader
    from rllm_trace_uploader.uploader import TraceUploader

    db_path = str(tmp_path / "traces.db")
    _make_fake_sqlite_db(db_path, n_rows=250)

    up = TraceUploader(
        db_path=db_path,
        state_path=str(tmp_path / "state.txt"),
        weave_project="test",
        sidecar=SidecarReader(str(tmp_path)),
    )
    rows = asyncio.run(up.fetch_new_traces(limit=100))
    assert len(rows) == 100
    assert rows[0].trace_id == "trace-0000"
    assert rows[-1].trace_id == "trace-0099"
    assert rows[0].session_id == "session-0"

    # Advance cursor manually (publish_batch does this normally); verify next page
    up._last_rowid = rows[-1].rowid
    rows2 = asyncio.run(up.fetch_new_traces(limit=100))
    assert len(rows2) == 100
    assert rows2[0].trace_id == "trace-0100"

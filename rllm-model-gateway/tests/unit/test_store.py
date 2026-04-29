"""Tests for MemoryTraceStore and SqliteTraceStore."""

import os
import tempfile

import pytest
from rllm_model_gateway.store.memory_store import MemoryTraceStore
from rllm_model_gateway.store.sqlite_store import SqliteTraceStore


# Parametrise to run every test against both store backends
@pytest.fixture(params=["memory", "sqlite"])
def store(request):
    if request.param == "memory":
        yield MemoryTraceStore()
    else:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield SqliteTraceStore(db_path=path)
        os.unlink(path)


class TestStoreTrace:
    @pytest.mark.asyncio
    async def test_store_and_get(self, store):
        await store.store_trace("t1", "s1", {"msg": "hello"})
        trace = await store.get_trace("t1")
        assert trace is not None
        assert trace["msg"] == "hello"

    @pytest.mark.asyncio
    async def test_get_missing(self, store):
        assert await store.get_trace("nonexistent") is None

    @pytest.mark.asyncio
    async def test_overwrite(self, store):
        await store.store_trace("t1", "s1", {"v": 1})
        await store.store_trace("t1", "s1", {"v": 2})
        trace = await store.get_trace("t1")
        assert trace["v"] == 2


class TestSessionTraces:
    @pytest.mark.asyncio
    async def test_get_session_traces(self, store):
        await store.store_trace("t1", "s1", {"order": 1})
        await store.store_trace("t2", "s1", {"order": 2})
        await store.store_trace("t3", "s2", {"order": 3})

        traces = await store.get_session_traces("s1")
        assert len(traces) == 2
        assert traces[0]["order"] == 1
        assert traces[1]["order"] == 2

    @pytest.mark.asyncio
    async def test_empty_session(self, store):
        traces = await store.get_session_traces("empty")
        assert traces == []

    @pytest.mark.asyncio
    async def test_limit(self, store):
        for i in range(5):
            await store.store_trace(f"t{i}", "s1", {"i": i})
        traces = await store.get_session_traces("s1", limit=3)
        assert len(traces) == 3


class TestDeleteSession:
    @pytest.mark.asyncio
    async def test_delete(self, store):
        await store.store_trace("t1", "s1", {"x": 1})
        await store.store_trace("t2", "s1", {"x": 2})

        deleted = await store.delete_session("s1")
        assert deleted == 2
        assert await store.get_session_traces("s1") == []

    @pytest.mark.asyncio
    async def test_delete_empty(self, store):
        deleted = await store.delete_session("nonexistent")
        assert deleted == 0


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list(self, store):
        await store.store_trace("t1", "s1", {})
        await store.store_trace("t2", "s1", {})
        await store.store_trace("t3", "s2", {})

        sessions = await store.list_sessions()
        assert len(sessions) == 2
        session_ids = {s["session_id"] for s in sessions}
        assert session_ids == {"s1", "s2"}

        s1 = next(s for s in sessions if s["session_id"] == "s1")
        assert s1["trace_count"] == 2

    @pytest.mark.asyncio
    async def test_list_limit(self, store):
        await store.store_trace("t1", "s1", {})
        await store.store_trace("t2", "s2", {})
        await store.store_trace("t3", "s3", {})

        sessions = await store.list_sessions(limit=2)
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_list_empty(self, store):
        sessions = await store.list_sessions()
        assert sessions == []


class TestFlush:
    @pytest.mark.asyncio
    async def test_flush_no_error(self, store):
        await store.flush()  # should not raise


class TestSqliteStoreSpecific:
    @pytest.mark.asyncio
    async def test_sqlite_store_uses_wal_mode(self, tmp_path):
        path = tmp_path / "gateway_traces.db"
        store = SqliteTraceStore(db_path=str(path))
        try:
            await store.store_trace("t1", "s1", {"msg": "hello"})
            conn = await store._get_conn()
            async with conn.execute("PRAGMA journal_mode") as cur:
                row = await cur.fetchone()
            assert row is not None
            assert row[0].lower() == "wal"
        finally:
            await store.close()


class TestRunIdScoping:
    """run_id tags + filters — the same session_id can coexist across runs."""

    @pytest.mark.asyncio
    async def test_get_session_traces_filtered_by_run_id(self, store):
        await store.store_trace("a1", "eval-0", {"v": "a1"}, run_id="run-A")
        await store.store_trace("a2", "eval-0", {"v": "a2"}, run_id="run-A")
        await store.store_trace("b1", "eval-0", {"v": "b1"}, run_id="run-B")
        await store.store_trace("b2", "eval-0", {"v": "b2"}, run_id="run-B")

        in_a = await store.get_session_traces("eval-0", run_id="run-A")
        assert {t["v"] for t in in_a} == {"a1", "a2"}
        in_b = await store.get_session_traces("eval-0", run_id="run-B")
        assert {t["v"] for t in in_b} == {"b1", "b2"}

    @pytest.mark.asyncio
    async def test_get_session_traces_cross_run_when_run_id_none(self, store):
        await store.store_trace("a1", "eval-0", {"v": "a1"}, run_id="run-A")
        await store.store_trace("b1", "eval-0", {"v": "b1"}, run_id="run-B")
        # run_id=None (default) → cross-run.
        traces = await store.get_session_traces("eval-0")
        assert {t["v"] for t in traces} == {"a1", "b1"}

    @pytest.mark.asyncio
    async def test_list_sessions_emits_run_id(self, store):
        await store.store_trace("a1", "eval-0", {}, run_id="run-A")
        await store.store_trace("b1", "eval-0", {}, run_id="run-B")
        sessions = await store.list_sessions()
        # Same session_id appears once per run_id.
        keys = {(s["session_id"], s["run_id"]) for s in sessions}
        assert keys == {("eval-0", "run-A"), ("eval-0", "run-B")}

    @pytest.mark.asyncio
    async def test_list_sessions_filtered_by_run_id(self, store):
        await store.store_trace("a1", "eval-0", {}, run_id="run-A")
        await store.store_trace("b1", "eval-1", {}, run_id="run-B")
        a = await store.list_sessions(run_id="run-A")
        assert {s["session_id"] for s in a} == {"eval-0"}
        b = await store.list_sessions(run_id="run-B")
        assert {s["session_id"] for s in b} == {"eval-1"}

    @pytest.mark.asyncio
    async def test_delete_session_run_scoped(self, store):
        await store.store_trace("a1", "eval-0", {}, run_id="run-A")
        await store.store_trace("b1", "eval-0", {}, run_id="run-B")

        # Delete only run-A's eval-0; run-B's eval-0 stays.
        deleted = await store.delete_session("eval-0", run_id="run-A")
        assert deleted == 1
        remaining = await store.get_session_traces("eval-0")
        assert len(remaining) == 1


class TestRunsTable:
    @pytest.mark.asyncio
    async def test_register_then_list(self, store):
        await store.register_run("r1", {"benchmark": "gsm8k", "model": "m"})
        runs = await store.list_runs()
        assert [r["run_id"] for r in runs] == ["r1"]
        assert runs[0]["metadata"] == {"benchmark": "gsm8k", "model": "m"}
        assert runs[0]["ended_at"] is None

    @pytest.mark.asyncio
    async def test_register_is_idempotent_and_clears_ended_at(self, store):
        await store.register_run("r1", {"v": 1})
        await store.end_run("r1")
        await store.register_run("r1", {"v": 2})  # re-register
        runs = await store.list_runs()
        assert runs[0]["metadata"] == {"v": 2}
        assert runs[0]["ended_at"] is None

    @pytest.mark.asyncio
    async def test_end_run_stamps_ended_at(self, store):
        await store.register_run("r1", {})
        await store.end_run("r1")
        runs = await store.list_runs()
        assert runs[0]["ended_at"] is not None

    @pytest.mark.asyncio
    async def test_runs_ordered_by_started_desc(self, store):
        import asyncio

        await store.register_run("r-old", {})
        await asyncio.sleep(0.01)
        await store.register_run("r-new", {})
        runs = await store.list_runs()
        assert [r["run_id"] for r in runs] == ["r-new", "r-old"]


class TestConcurrentWriters:
    """Two ``SqliteTraceStore`` instances writing the same db with different run_ids.

    Mirrors the production case: two ``rllm eval`` invocations each boot
    their own gateway pointing at ``~/.rllm/gateway/traces.db``. WAL +
    busy_timeout serialises the writes; we just need to confirm no rows
    are lost and per-run filtering still works.
    """

    @pytest.mark.asyncio
    async def test_two_stores_share_one_db(self, tmp_path):
        import asyncio

        path = str(tmp_path / "shared.db")
        store_a = SqliteTraceStore(db_path=path)
        store_b = SqliteTraceStore(db_path=path)
        try:
            await asyncio.gather(
                store_a.register_run("run-A", {"who": "A"}),
                store_b.register_run("run-B", {"who": "B"}),
            )

            async def write_n(s, run_id, n, prefix):
                for i in range(n):
                    await s.store_trace(f"{prefix}{i}", "eval-0", {"i": i}, run_id=run_id)

            # Interleave two writers; sqlite WAL+busy_timeout should
            # serialise them without error.
            await asyncio.gather(
                write_n(store_a, "run-A", 20, "a"),
                write_n(store_b, "run-B", 20, "b"),
            )

            # Either store can read both runs' data — they share the file.
            sessions = await store_a.list_sessions()
            keys = {(s["session_id"], s["run_id"]) for s in sessions}
            assert keys == {("eval-0", "run-A"), ("eval-0", "run-B")}
            for s in sessions:
                assert s["trace_count"] == 20

            runs = await store_b.list_runs()
            assert {r["run_id"] for r in runs} == {"run-A", "run-B"}
        finally:
            await store_a.close()
            await store_b.close()

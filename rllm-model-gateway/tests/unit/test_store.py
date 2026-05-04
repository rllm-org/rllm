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


def _trace(model="gpt-x", harness=None, latency_ms=10, finish_reason="stop", step_id=None, **extra):
    """Build a TraceRecord-shaped dict for query_traces tests."""
    return {
        "model": model,
        "harness": harness,
        "latency_ms": latency_ms,
        "finish_reason": finish_reason,
        "response_message": {"role": "assistant", "content": "hi"},
        "step_id": step_id,
        **extra,
    }


class TestQueryTraces:
    @pytest.mark.asyncio
    async def test_filter_by_run_id(self, store):
        await store.store_trace("a1", "s1", _trace(), run_id="run-A")
        await store.store_trace("a2", "s1", _trace(), run_id="run-A")
        await store.store_trace("b1", "s1", _trace(), run_id="run-B")

        rows = await store.query_traces(run_id="run-A")
        assert len(rows) == 2
        rows = await store.query_traces(run_id="run-B")
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_filter_by_session(self, store):
        await store.store_trace("t1", "s1", _trace(), run_id="r")
        await store.store_trace("t2", "s2", _trace(), run_id="r")
        rows = await store.query_traces(session_id="s1")
        assert len(rows) == 1
        rows = await store.query_traces(session_id="s2")
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_filter_by_model(self, store):
        await store.store_trace("t1", "s", _trace(model="gpt-5"), run_id="r")
        await store.store_trace("t2", "s", _trace(model="claude-4-7"), run_id="r")
        rows = await store.query_traces(model="claude-4-7")
        assert len(rows) == 1
        assert rows[0]["model"] == "claude-4-7"

    @pytest.mark.asyncio
    async def test_filter_by_harness(self, store):
        await store.store_trace("t1", "s", _trace(harness="bash"), run_id="r")
        await store.store_trace("t2", "s", _trace(harness="claude-code"), run_id="r")
        await store.store_trace("t3", "s", _trace(harness=None), run_id="r")
        rows = await store.query_traces(harness="bash")
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_filter_by_has_error(self, store):
        # Healthy completion.
        await store.store_trace("ok", "s", _trace(finish_reason="stop"), run_id="r")
        # Empty response, no finish reason → has_error.
        await store.store_trace(
            "err",
            "s",
            {"model": "x", "latency_ms": 5, "finish_reason": None, "response_message": {}},
            run_id="r",
        )
        # Upstream error in raw_response → has_error.
        await store.store_trace(
            "err2",
            "s",
            {
                "model": "x",
                "latency_ms": 0,
                "finish_reason": "stop",
                "response_message": {"role": "assistant", "content": ""},
                "raw_response": {"error": {"message": "rate limit"}},
            },
            run_id="r",
        )

        ok = await store.query_traces(has_error=False)
        err = await store.query_traces(has_error=True)
        assert len(ok) == 1
        assert len(err) == 2

    @pytest.mark.asyncio
    async def test_filter_by_latency_range(self, store):
        await store.store_trace("fast", "s", _trace(latency_ms=10), run_id="r")
        await store.store_trace("med", "s", _trace(latency_ms=100), run_id="r")
        await store.store_trace("slow", "s", _trace(latency_ms=1000), run_id="r")

        rows = await store.query_traces(latency_min=50)
        assert len(rows) == 2
        rows = await store.query_traces(latency_max=500)
        assert len(rows) == 2
        rows = await store.query_traces(latency_min=50, latency_max=500)
        assert len(rows) == 1
        assert rows[0]["latency_ms"] == 100

    @pytest.mark.asyncio
    async def test_combined_filters(self, store):
        await store.store_trace("a", "s1", _trace(model="gpt", harness="bash"), run_id="run-A")
        await store.store_trace("b", "s1", _trace(model="claude", harness="bash"), run_id="run-A")
        await store.store_trace("c", "s2", _trace(model="gpt", harness="bash"), run_id="run-B")

        rows = await store.query_traces(run_id="run-A", model="gpt")
        assert len(rows) == 1
        rows = await store.query_traces(run_id="run-A", harness="bash")
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_pagination_cursor(self, store):
        import asyncio

        for i in range(10):
            await store.store_trace(f"t{i}", "s", _trace(), run_id="r")
            await asyncio.sleep(0.001)  # ensure distinct created_at

        # Newest-first feed, paged via `until`.
        page1 = await store.query_traces(limit=4, order="DESC")
        assert len(page1) == 4
        cursor = page1[-1]["_created_at"]
        page2 = await store.query_traces(limit=4, until=cursor, order="DESC")
        assert len(page2) == 4
        # No overlap between pages.
        ids1 = {t.get("trace_id") for t in page1}
        ids2 = {t.get("trace_id") for t in page2}
        assert ids1.isdisjoint(ids2) or (ids1 == {None} and ids2 == {None})

    @pytest.mark.asyncio
    async def test_live_tail_cursor(self, store):
        await store.store_trace("t1", "s", _trace(), run_id="r")
        rows = await store.query_traces(order="ASC")
        cursor = rows[-1]["_created_at"]
        # No new traces yet — cursor should return nothing.
        assert await store.query_traces(since=cursor, order="ASC") == []
        # Add another and tail.
        await store.store_trace("t2", "s", _trace(), run_id="r")
        new_rows = await store.query_traces(since=cursor, order="ASC")
        assert len(new_rows) == 1

    @pytest.mark.asyncio
    async def test_order(self, store):
        import asyncio

        for i in range(5):
            await store.store_trace(f"t{i}", "s", _trace(latency_ms=i), run_id="r")
            await asyncio.sleep(0.001)

        asc = await store.query_traces(order="ASC")
        desc = await store.query_traces(order="DESC")
        assert [r["latency_ms"] for r in asc] == [0, 1, 2, 3, 4]
        assert [r["latency_ms"] for r in desc] == [4, 3, 2, 1, 0]

    @pytest.mark.asyncio
    async def test_invalid_order(self, store):
        with pytest.raises(ValueError):
            await store.query_traces(order="bogus")


class TestCountTraces:
    @pytest.mark.asyncio
    async def test_count_with_filters(self, store):
        await store.store_trace("a", "s1", _trace(model="gpt"), run_id="r")
        await store.store_trace("b", "s1", _trace(model="gpt"), run_id="r")
        await store.store_trace("c", "s2", _trace(model="claude"), run_id="r")

        assert await store.count_traces() == 3
        assert await store.count_traces(model="gpt") == 2
        assert await store.count_traces(session_id="s2") == 1
        assert await store.count_traces(model="bogus") == 0


class TestFacets:
    @pytest.mark.asyncio
    async def test_facets(self, store):
        await store.store_trace("a", "s1", _trace(model="gpt-5", harness="bash"), run_id="run-A")
        await store.store_trace("b", "s1", _trace(model="claude", harness="bash"), run_id="run-A")
        await store.store_trace("c", "s2", _trace(model="gpt-5", harness="claude-code"), run_id="run-B")

        f = await store.facets()
        assert f["models"] == ["claude", "gpt-5"]
        assert f["harnesses"] == ["bash", "claude-code"]
        assert f["runs"] == ["run-A", "run-B"]

    @pytest.mark.asyncio
    async def test_facets_excludes_empty(self, store):
        await store.store_trace("a", "s", _trace(model="", harness=None), run_id="")
        f = await store.facets()
        assert f["models"] == []
        assert f["harnesses"] == []
        assert f["runs"] == []


class TestSchemaMigration:
    """Drop-and-rebuild on schema-version mismatch."""

    @pytest.mark.asyncio
    async def test_rebuild_on_schema_bump(self, tmp_path):
        # Plant a v1-shaped db with a row in it.
        path = str(tmp_path / "old.db")
        import aiosqlite

        async with aiosqlite.connect(path) as conn:
            await conn.execute("PRAGMA user_version = 1")
            await conn.execute("CREATE TABLE traces (trace_id TEXT PRIMARY KEY, data TEXT, created_at REAL)")
            await conn.execute("INSERT INTO traces VALUES ('legacy', '{}', 1.0)")
            await conn.commit()

        # Opening with the new store drops & rebuilds.
        store = SqliteTraceStore(db_path=path)
        try:
            await store.store_trace("fresh", "s", _trace(), run_id="r")
            # Legacy row gone.
            assert await store.get_trace("legacy") is None
            assert await store.get_trace("fresh") is not None
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_rebuild_on_pre_versioning_legacy_db(self, tmp_path):
        """Legacy dbs predating user_version stamping (current==0) still rebuild.

        Reproduces the in-the-wild error where existing
        ``~/.rllm/gateway/traces.db`` had ``PRAGMA user_version=0`` and
        a v1 ``traces`` table with no ``run_id`` column. The original
        guard ``if current and current != _SCHEMA_VERSION`` skipped the
        drop because ``current == 0`` is falsy, then the v2 indexes
        failed against the v1 columns.
        """
        path = str(tmp_path / "unversioned.db")
        import aiosqlite

        async with aiosqlite.connect(path) as conn:
            # Note: do NOT stamp user_version — defaults to 0.
            await conn.execute("CREATE TABLE traces (trace_id TEXT PRIMARY KEY, data TEXT, created_at REAL)")
            await conn.execute("CREATE TABLE trace_sessions (trace_id TEXT, session_id TEXT, created_at REAL, PRIMARY KEY (trace_id, session_id))")
            await conn.execute("INSERT INTO traces VALUES ('legacy', '{}', 1.0)")
            await conn.commit()

        store = SqliteTraceStore(db_path=path)
        try:
            await store.store_trace("fresh", "s", _trace(), run_id="r")
            # Legacy row gone, v2 columns work.
            assert await store.get_trace("legacy") is None
            rows = await store.query_traces(run_id="r")
            assert len(rows) == 1
        finally:
            await store.close()


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

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

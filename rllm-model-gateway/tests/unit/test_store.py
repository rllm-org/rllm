"""Tests for MemoryTraceStore and SqliteTraceStore."""

import array
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


class TestTokenArrayPacking:
    """The memory store packs token-id/logprob lists into array.array to save
    RAM, but the packing must be invisible to readers (plain lists back out)."""

    # Ids above CPython's small-int intern cap and a realistic vocab range.
    _TRACE = {
        "prompt_token_ids": [1000, 200000, 42, 1000, 151000],
        "completion_token_ids": [12, 99999],
        "logprobs": [-0.5, -12.3456, -0.0001],
        "messages": [{"role": "user", "content": "hi"}],
    }

    @pytest.mark.asyncio
    async def test_roundtrip_transparent(self, store):
        """Both backends return plain lists with identical values."""
        await store.store_trace("t1", "s1", dict(self._TRACE))
        for getter in (
            await store.get_trace("t1"),
            (await store.get_session_traces("s1"))[0],
        ):
            for field in ("prompt_token_ids", "completion_token_ids", "logprobs"):
                assert isinstance(getter[field], list)
                assert getter[field] == self._TRACE[field]
            assert getter["messages"] == self._TRACE["messages"]

    @pytest.mark.asyncio
    async def test_memory_store_packs_internally(self):
        mem = MemoryTraceStore()
        await mem.store_trace("t1", "s1", dict(self._TRACE))
        stored = mem._traces["t1"]
        assert isinstance(stored["prompt_token_ids"], array.array)
        assert isinstance(stored["completion_token_ids"], array.array)
        assert isinstance(stored["logprobs"], array.array)
        assert stored["prompt_token_ids"].typecode == "i"
        assert stored["logprobs"].typecode == "d"
        # messages and other fields are untouched
        assert stored["messages"] == self._TRACE["messages"]

    @pytest.mark.asyncio
    async def test_pack_does_not_mutate_caller_dict(self):
        mem = MemoryTraceStore()
        data = dict(self._TRACE)
        await mem.store_trace("t1", "s1", data)
        assert isinstance(data["prompt_token_ids"], list)  # caller's copy intact

    @pytest.mark.asyncio
    async def test_non_numeric_fields_fall_back(self):
        mem = MemoryTraceStore()
        # None logprobs, empty ids, and a None element must not raise or corrupt.
        data = {
            "prompt_token_ids": [],
            "completion_token_ids": [1, 2, 3],
            "logprobs": None,
        }
        await mem.store_trace("t1", "s1", data)
        got = await mem.get_trace("t1")
        assert got["prompt_token_ids"] == []
        assert got["completion_token_ids"] == [1, 2, 3]
        assert got["logprobs"] is None

        data2 = {"logprobs": [-0.1, None, -0.3]}  # ragged -> keep as list
        await mem.store_trace("t2", "s1", data2)
        assert isinstance(mem._traces["t2"]["logprobs"], list)
        assert (await mem.get_trace("t2"))["logprobs"] == [-0.1, None, -0.3]


class TestTombstone:
    """A trace that lands after its session was deleted must NOT resurrect the
    session (the fire-and-forget persist race that leaked orphaned sessions).
    Session ids are unique and never reused, so the block is permanent."""

    @pytest.mark.asyncio
    async def test_straggler_write_after_delete_is_dropped(self):
        mem = MemoryTraceStore()
        await mem.store_trace("t1", "s1", {"prompt_token_ids": [1, 2, 3]})
        assert await mem.delete_session("s1") == 1
        # straggler completion for the same session lands after the delete
        await mem.store_trace("t2", "s1", {"prompt_token_ids": [4, 5, 6]})
        assert await mem.get_session_traces("s1") == []          # not resurrected
        assert await mem.get_trace("t2") is None
        assert all(s["session_id"] != "s1" for s in await mem.list_sessions())

    @pytest.mark.asyncio
    async def test_block_is_permanent(self):
        mem = MemoryTraceStore()
        await mem.delete_session("s1")
        for tid in ("t1", "t2", "t3"):                            # repeated late writes
            await mem.store_trace(tid, "s1", {})
        assert await mem.get_session_traces("s1") == []

    @pytest.mark.asyncio
    async def test_delete_does_not_block_other_sessions(self):
        mem = MemoryTraceStore()
        await mem.store_trace("t1", "s1", {})
        await mem.delete_session("s1")
        await mem.store_trace("t2", "s2", {})                     # unrelated session
        assert len(await mem.get_session_traces("s2")) == 1

    @pytest.mark.asyncio
    async def test_cap_evicts_oldest_tombstone(self):
        mem = MemoryTraceStore(max_tombstones=2)
        for sid in ("s1", "s2", "s3"):                            # s1 evicted (FIFO)
            await mem.delete_session(sid)
        assert list(mem._tombstones) == ["s2", "s3"]
        await mem.store_trace("t2", "s2", {})                     # still blocked
        assert await mem.get_session_traces("s2") == []
        await mem.store_trace("t1", "s1", {})                     # evicted past cap -> allowed
        assert len(await mem.get_session_traces("s1")) == 1

    @pytest.mark.asyncio
    async def test_zero_cap_disables_tombstone(self):
        mem = MemoryTraceStore(max_tombstones=0)
        await mem.store_trace("t1", "s1", {})
        await mem.delete_session("s1")
        assert mem._tombstones == {}                             # nothing remembered
        await mem.store_trace("t2", "s1", {})                     # legacy resurrection behaviour
        assert len(await mem.get_session_traces("s1")) == 1


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

    def test_explicit_db_path_does_not_warn(self, tmp_path, caplog):
        path = tmp_path / "explicit.db"
        with caplog.at_level("WARNING", logger="rllm_model_gateway.store.sqlite_store"):
            store = SqliteTraceStore(db_path=str(path))
        assert store.db_path == str(path)
        assert not any("db_path not set" in rec.message for rec in caplog.records)

    def test_missing_db_path_warns_and_resolves(self, tmp_path, monkeypatch, caplog):
        # Redirect ~/.rllm into tmp_path so the test doesn't touch the user's home.
        monkeypatch.setenv("HOME", str(tmp_path))
        with caplog.at_level("WARNING", logger="rllm_model_gateway.store.sqlite_store"):
            store = SqliteTraceStore(db_path=None)
        assert store.db_path  # auto-resolved to a non-empty path
        assert store.db_path.endswith("gateway_traces.db")
        warnings = [rec for rec in caplog.records if "db_path not set" in rec.message]
        assert len(warnings) == 1
        assert store.db_path in warnings[0].message

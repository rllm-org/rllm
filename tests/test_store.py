"""Tests for the cross-episode Store interface and InMemoryStore implementation."""

from __future__ import annotations

import asyncio

import pytest

from rllm.workflows.store import InMemoryStore, Store

# ---------------------------------------------------------------------------
# InMemoryStore — basic CRUD
# ---------------------------------------------------------------------------


@pytest.fixture()
def store():
    return InMemoryStore()


def _run(coro):
    """Helper to run an async coroutine in a fresh event loop."""
    return asyncio.run(coro)


class TestInMemoryStoreCRUD:
    def test_get_missing_key_returns_default(self, store):
        assert _run(store.get("no_such_key")) is None
        assert _run(store.get("no_such_key", "fallback")) == "fallback"

    def test_set_and_get(self, store):
        _run(store.set("k", "v"))
        assert _run(store.get("k")) == "v"

    def test_set_overwrites(self, store):
        _run(store.set("k", "v1"))
        _run(store.set("k", "v2"))
        assert _run(store.get("k")) == "v2"

    def test_delete_existing_key(self, store):
        _run(store.set("k", 1))
        assert _run(store.delete("k")) is True
        assert _run(store.get("k")) is None

    def test_delete_missing_key(self, store):
        assert _run(store.delete("nope")) is False

    def test_keys(self, store):
        _run(store.set("a", 1))
        _run(store.set("b", 2))
        assert sorted(_run(store.keys())) == ["a", "b"]

    def test_has(self, store):
        assert _run(store.has("x")) is False
        _run(store.set("x", 42))
        assert _run(store.has("x")) is True

    def test_clear(self, store):
        _run(store.set("a", 1))
        _run(store.set("b", 2))
        _run(store.clear())
        assert _run(store.keys()) == []

    def test_stores_arbitrary_types(self, store):
        _run(store.set("list", [1, 2, 3]))
        _run(store.set("dict", {"nested": True}))
        assert _run(store.get("list")) == [1, 2, 3]
        assert _run(store.get("dict")) == {"nested": True}


# ---------------------------------------------------------------------------
# Concurrent access
# ---------------------------------------------------------------------------


class TestConcurrentAccess:
    def test_concurrent_writes_no_corruption(self, store):
        """Many coroutines incrementing a counter should not lose updates."""

        async def _increment(s: InMemoryStore, n: int):
            for _ in range(n):
                val = await s.get("counter", 0)
                await s.set("counter", val + 1)

        async def _main():
            n_tasks, n_increments = 10, 50
            await asyncio.gather(*[_increment(store, n_increments) for _ in range(n_tasks)])
            return await store.get("counter")

        # Because asyncio.Lock serialises access, no increments are lost.
        result = _run(_main())
        assert result == 10 * 50

    def test_concurrent_reads_during_writes(self, store):
        """Reads should never see a partially updated store."""

        async def _writer(s: InMemoryStore):
            for i in range(100):
                await s.set("val", i)

        async def _reader(s: InMemoryStore, results: list):
            for _ in range(100):
                v = await s.get("val")
                if v is not None:
                    results.append(v)

        async def _main():
            results: list[int] = []
            await asyncio.gather(_writer(store), _reader(store, results))
            # Every read value should be an integer (not a corrupted type)
            assert all(isinstance(v, int) for v in results)

        _run(_main())


# ---------------------------------------------------------------------------
# Store shared across simulated workflow instances
# ---------------------------------------------------------------------------


class TestStoreSharing:
    def test_two_instances_share_state(self, store):
        """Two 'workflow instances' holding the same store see each other's writes."""

        async def _workflow_a(s: Store):
            await s.set("improved_prompt", "version_1")

        async def _workflow_b(s: Store):
            return await s.get("improved_prompt")

        async def _main():
            await _workflow_a(store)
            result = await _workflow_b(store)
            return result

        assert _run(_main()) == "version_1"

    def test_interleaved_read_write(self, store):
        """Simulates the ERL cross-episode memory pattern: write on success, read on next episode."""

        async def _run_episode(s: Store, task_id: int, succeed: bool):
            prompt = await s.get("prompt", "initial")
            # On success, store the improved prompt for next episodes.
            if succeed:
                await s.set("prompt", f"improved_by_task_{task_id}")
            return prompt

        async def _main():
            # Episode 0 succeeds and stores an improved prompt.
            p0 = await _run_episode(store, 0, succeed=True)
            assert p0 == "initial"

            # Episode 1 reads the improved prompt.
            p1 = await _run_episode(store, 1, succeed=False)
            assert p1 == "improved_by_task_0"

            # Episode 2 also reads the same improved prompt (no new write in episode 1).
            p2 = await _run_episode(store, 2, succeed=True)
            assert p2 == "improved_by_task_0"

            # Episode 3 reads the prompt from episode 2's success.
            p3 = await _run_episode(store, 3, succeed=False)
            assert p3 == "improved_by_task_2"

        _run(_main())


# ---------------------------------------------------------------------------
# Workflow integration (lightweight, no GPU)
# ---------------------------------------------------------------------------


class TestWorkflowIntegration:
    def test_workflow_receives_store(self):
        """Workflow.__init__ accepts and exposes the store attribute."""
        from unittest.mock import MagicMock

        from rllm.workflows.workflow import Workflow

        s = InMemoryStore()
        mock_engine = MagicMock()
        mock_executor = MagicMock()

        # Create a minimal concrete subclass.
        class DummyWorkflow(Workflow):
            async def run(self, task, uid, **kwargs):
                return None

        wf = DummyWorkflow(rollout_engine=mock_engine, executor=mock_executor, store=s)
        assert wf.store is s

    def test_workflow_without_store(self):
        """Workflow.store defaults to None when not provided."""
        from unittest.mock import MagicMock

        from rllm.workflows.workflow import Workflow

        mock_engine = MagicMock()
        mock_executor = MagicMock()

        class DummyWorkflow(Workflow):
            async def run(self, task, uid, **kwargs):
                return None

        wf = DummyWorkflow(rollout_engine=mock_engine, executor=mock_executor)
        assert wf.store is None

"""Tests for SessionRouter and routing policies."""

from unittest.mock import AsyncMock, patch

import pytest
from rllm_model_gateway.models import WorkerInfo
from rllm_model_gateway.session_router import (
    SessionRouter,
    StickyLeastLoadedPolicy,
)


def _w(wid: str, url: str) -> WorkerInfo:
    return WorkerInfo(worker_id=wid, url=url)


class TestStickyLeastLoadedPolicy:
    def test_new_session_picks_least_loaded(self):
        policy = StickyLeastLoadedPolicy()
        workers = [_w("w1", "http://w1"), _w("w2", "http://w2")]
        active = {"http://w1": 5, "http://w2": 2}

        selected = policy.select_worker(workers, "sess-1", active)
        assert selected.url == "http://w2"

    def test_sticky_returns_cached(self):
        policy = StickyLeastLoadedPolicy()
        workers = [_w("w1", "http://w1"), _w("w2", "http://w2")]
        active = {"http://w1": 0, "http://w2": 0}

        first = policy.select_worker(workers, "sess-1", active)
        # Even if load changes, same session returns same worker
        active[first.url] = 100
        second = policy.select_worker(workers, "sess-1", active)
        assert first.url == second.url

    def test_sticky_fallback_if_worker_gone(self):
        policy = StickyLeastLoadedPolicy()
        workers = [_w("w1", "http://w1"), _w("w2", "http://w2")]

        policy.select_worker(workers, "sess-1", {})
        # Remove w1 from available workers
        new_workers = [_w("w2", "http://w2")]
        selected = policy.select_worker(new_workers, "sess-1", {})
        assert selected.url == "http://w2"

    def test_no_session_id_picks_least_loaded(self):
        policy = StickyLeastLoadedPolicy()
        workers = [_w("w1", "http://w1"), _w("w2", "http://w2")]
        active = {"http://w1": 3, "http://w2": 1}

        selected = policy.select_worker(workers, None, active)
        assert selected.url == "http://w2"

    def test_on_worker_change_clears_cache(self):
        policy = StickyLeastLoadedPolicy()
        workers = [_w("w1", "http://w1"), _w("w2", "http://w2")]

        policy.select_worker(workers, "sess-1", {})
        policy.on_worker_change(workers)
        # After cache clear, session is no longer sticky
        assert policy._cache.get("sess-1") is None


class TestSessionRouter:
    def test_route_and_release(self):
        router = SessionRouter()
        router.add_worker(_w("w1", "http://w1"))

        worker = router.route("sess-1")
        assert worker.url == "http://w1"
        assert router.active_counts["http://w1"] == 1

        router.release("http://w1")
        assert router.active_counts["http://w1"] == 0

    def test_no_workers_raises(self):
        router = SessionRouter()
        with pytest.raises(RuntimeError, match="No healthy workers"):
            router.route("sess-1")

    def test_all_dead_raises(self):
        router = SessionRouter()
        router.add_worker(_w("w1", "http://w1"))
        router.dead_workers.add("http://w1")

        with pytest.raises(RuntimeError, match="No healthy workers"):
            router.route("sess-1")

    def test_dead_worker_excluded(self):
        router = SessionRouter()
        router.add_worker(_w("w1", "http://w1"))
        router.add_worker(_w("w2", "http://w2"))
        router.dead_workers.add("http://w1")

        worker = router.route("sess-1")
        assert worker.url == "http://w2"

    def test_add_worker_idempotent(self):
        router = SessionRouter()
        router.add_worker(_w("w1", "http://w1"))
        router.add_worker(_w("w1", "http://w1"))
        assert len(router.workers) == 1

    def test_remove_worker(self):
        router = SessionRouter()
        router.add_worker(_w("w1", "http://w1"))
        router.add_worker(_w("w2", "http://w2"))

        router.remove_worker("http://w1")
        assert len(router.workers) == 1
        assert router.workers[0].url == "http://w2"

    def test_get_workers_includes_health(self):
        router = SessionRouter()
        router.add_worker(_w("w1", "http://w1"))
        router.add_worker(_w("w2", "http://w2"))
        router.dead_workers.add("http://w2")

        workers = router.get_workers()
        assert len(workers) == 2
        w1 = next(w for w in workers if w.worker_id == "w1")
        w2 = next(w for w in workers if w.worker_id == "w2")
        assert w1.healthy is True
        assert w2.healthy is False

    def test_sticky_routing_across_calls(self):
        router = SessionRouter()
        router.add_worker(_w("w1", "http://w1"))
        router.add_worker(_w("w2", "http://w2"))

        first = router.route("sess-1")
        router.release(first.url)
        second = router.route("sess-1")
        router.release(second.url)

        assert first.url == second.url

    def test_release_clamps_to_zero(self):
        router = SessionRouter()
        router.add_worker(_w("w1", "http://w1"))
        router.release("http://w1")
        assert router.active_counts["http://w1"] == 0


class TestHealthLoop:
    """Tests for the health check loop, including dead worker recovery."""

    @pytest.mark.asyncio
    async def test_worker_marked_dead_after_threshold(self):
        """Worker should be marked dead after consecutive health check failures."""
        router = SessionRouter(health_check_interval=0.01, failure_threshold=2)
        router.add_worker(_w("w1", "http://w1"))

        # Mock _check to always fail
        with patch.object(router, "_check", new_callable=AsyncMock, return_value=("http://w1", False)):
            await router.start_health_checks()
            # Wait enough for a couple of health loop iterations
            import asyncio

            await asyncio.sleep(0.1)
            await router.stop_health_checks()

        assert "http://w1" in router.dead_workers

    @pytest.mark.asyncio
    async def test_dead_worker_recovers(self):
        """A dead worker should be resurrected when health check passes."""
        router = SessionRouter(health_check_interval=0.01, failure_threshold=2)
        router.add_worker(_w("w1", "http://w1"))
        # Pre-mark as dead
        router.dead_workers.add("http://w1")
        router.failure_counts["http://w1"] = 5

        # Mock _check to succeed
        with patch.object(router, "_check", new_callable=AsyncMock, return_value=("http://w1", True)):
            await router.start_health_checks()
            import asyncio

            await asyncio.sleep(0.1)
            await router.stop_health_checks()

        assert "http://w1" not in router.dead_workers
        assert router.failure_counts["http://w1"] == 0

    @pytest.mark.asyncio
    async def test_healthy_worker_not_affected(self):
        """A healthy worker passing checks should stay healthy with zero failures."""
        router = SessionRouter(health_check_interval=0.01, failure_threshold=3)
        router.add_worker(_w("w1", "http://w1"))

        with patch.object(router, "_check", new_callable=AsyncMock, return_value=("http://w1", True)):
            await router.start_health_checks()
            import asyncio

            await asyncio.sleep(0.1)
            await router.stop_health_checks()

        assert "http://w1" not in router.dead_workers
        assert router.failure_counts["http://w1"] == 0

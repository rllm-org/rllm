"""Session-sticky worker routing with pluggable policies.

Reference implementations:
- verl ``AsyncLLMServerManager._choose_server()`` — LRU cache for session stickiness
- miles ``MilesRouter._use_url()`` / ``_finish_url()`` — least-loaded selection
"""

import asyncio
import logging
from collections import OrderedDict
from typing import Protocol

import httpx

from rllm_model_gateway.models import WorkerInfo

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# RoutingPolicy protocol
# ------------------------------------------------------------------


class RoutingPolicy(Protocol):
    """Pluggable worker selection strategy."""

    def select_worker(
        self,
        workers: list[WorkerInfo],
        session_id: str | None,
        active_counts: dict[str, int],
    ) -> WorkerInfo: ...

    def on_worker_change(self, workers: list[WorkerInfo]) -> None: ...


# ------------------------------------------------------------------
# Default policy: sticky sessions with least-loaded fallback
# ------------------------------------------------------------------


class LRUCache:
    """Minimal LRU cache backed by ``OrderedDict``."""

    def __init__(self, maxsize: int = 10_000) -> None:
        self._data: OrderedDict[str, str] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> str | None:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: str, value: str) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def clear(self) -> None:
        self._data.clear()


class StickyLeastLoadedPolicy:
    """Sticky sessions with least-loaded fallback.

    Mirrors verl ``AsyncLLMServerManager._choose_server()`` (LRU cache mapping
    session_id → worker) with a least-loaded tiebreaker for new sessions.
    """

    def __init__(self, max_cache_size: int = 10_000) -> None:
        self._cache = LRUCache(maxsize=max_cache_size)

    def select_worker(
        self,
        workers: list[WorkerInfo],
        session_id: str | None,
        active_counts: dict[str, int],
    ) -> WorkerInfo:
        worker_urls = {w.url for w in workers}

        # Sticky: return cached worker if still alive
        if session_id:
            cached_url = self._cache.get(session_id)
            if cached_url and cached_url in worker_urls:
                return next(w for w in workers if w.url == cached_url)

        # Fallback: least-loaded
        worker = min(workers, key=lambda w: active_counts.get(w.url, 0))
        if session_id:
            self._cache.put(session_id, worker.url)
        return worker

    def on_worker_change(self, workers: list[WorkerInfo]) -> None:
        self._cache.clear()


# ------------------------------------------------------------------
# SessionRouter — manages worker pool + delegates to policy
# ------------------------------------------------------------------


class SessionRouter:
    """Worker pool manager with pluggable routing policy.

    Reference: miles ``MilesRouter._use_url()`` / ``_finish_url()``
    (``miles/router/router.py`` lines 214-235).
    """

    def __init__(
        self,
        policy: RoutingPolicy | None = None,
        health_check_interval: float = 10.0,
        failure_threshold: int = 3,
    ) -> None:
        self.workers: list[WorkerInfo] = []
        self.dead_workers: set[str] = set()
        self.active_counts: dict[str, int] = {}
        self.failure_counts: dict[str, int] = {}
        self.policy: RoutingPolicy = policy or StickyLeastLoadedPolicy()
        self._health_interval = health_check_interval
        self._failure_threshold = failure_threshold
        self._health_task: asyncio.Task[None] | None = None
        self._http: httpx.AsyncClient | None = None

    # -- Worker management -------------------------------------------------

    def add_worker(self, worker: WorkerInfo) -> None:
        if any(w.url == worker.url for w in self.workers):
            return
        self.workers.append(worker)
        self.active_counts.setdefault(worker.url, 0)
        self.failure_counts.setdefault(worker.url, 0)
        self.dead_workers.discard(worker.url)
        self.policy.on_worker_change(self.workers)

    def remove_worker(self, worker_url: str) -> None:
        self.workers = [w for w in self.workers if w.url != worker_url]
        self.active_counts.pop(worker_url, None)
        self.failure_counts.pop(worker_url, None)
        self.dead_workers.discard(worker_url)
        self.policy.on_worker_change(self.workers)

    def get_workers(self) -> list[WorkerInfo]:
        result: list[WorkerInfo] = []
        for w in self.workers:
            result.append(
                WorkerInfo(
                    worker_id=w.worker_id,
                    url=w.url,
                    model_name=w.model_name,
                    weight=w.weight,
                    healthy=w.url not in self.dead_workers,
                    active_requests=self.active_counts.get(w.url, 0),
                )
            )
        return result

    # -- Routing -----------------------------------------------------------

    def route(self, session_id: str | None = None) -> WorkerInfo:
        healthy = [w for w in self.workers if w.url not in self.dead_workers]
        if not healthy:
            raise RuntimeError("No healthy workers available")
        worker = self.policy.select_worker(healthy, session_id, self.active_counts)
        self.active_counts[worker.url] = self.active_counts.get(worker.url, 0) + 1
        return worker

    def release(self, worker_url: str) -> None:
        self.active_counts[worker_url] = max(0, self.active_counts.get(worker_url, 0) - 1)

    # -- Health checks -----------------------------------------------------

    async def start_health_checks(self) -> None:
        if self._health_task is not None:
            return
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        self._health_task = asyncio.create_task(self._health_loop())

    async def stop_health_checks(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _health_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._health_interval)
                # Check healthy workers
                alive_urls = [w.url for w in self.workers if w.url not in self.dead_workers]
                # Also check dead workers so they can recover
                dead_urls = [w.url for w in self.workers if w.url in self.dead_workers]
                all_urls = alive_urls + dead_urls
                if not all_urls:
                    continue

                results = await asyncio.gather(*(self._check(u) for u in all_urls), return_exceptions=True)
                for url, ok in results:
                    if not ok:
                        count = self.failure_counts.get(url, 0) + 1
                        self.failure_counts[url] = count
                        if count >= self._failure_threshold:
                            if url not in self.dead_workers:
                                logger.warning(
                                    "Worker %s failed %d health checks — marking dead",
                                    url,
                                    count,
                                )
                            self.dead_workers.add(url)
                    else:
                        if url in self.dead_workers:
                            logger.info("Worker %s recovered — marking healthy", url)
                            self.dead_workers.discard(url)
                        self.failure_counts[url] = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Health check loop error")
                await asyncio.sleep(5)

    async def _check(self, url: str) -> tuple[str, bool]:
        assert self._http is not None
        try:
            resp = await self._http.get(f"{url}/health")
            return url, resp.status_code == 200
        except Exception:
            return url, False

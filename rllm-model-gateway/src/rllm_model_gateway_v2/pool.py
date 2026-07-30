import asyncio
import logging
import multiprocessing
import queue
import threading
import time
import uuid
from multiprocessing.process import BaseProcess
from typing import Any

from rllm_model_gateway_v2.config import GatewayConfig
from rllm_model_gateway_v2.errors import GatewayError, WorkerUnavailableError
from rllm_model_gateway_v2.worker import worker_main

logger = logging.getLogger(__name__)


class WorkerPool:
    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._context = multiprocessing.get_context("spawn")
        self._request_queues: list[Any] = []
        self._response_queue: Any = None
        self._processes: list[BaseProcess] = []
        self._pending: dict[str, tuple[int, str | None, str, asyncio.AbstractEventLoop, asyncio.Future[Any]]] = {}
        self._pending_lock = threading.Lock()
        self._dispatcher: threading.Thread | None = None
        self._session_owners: dict[str, int] = {}
        self._active_sessions: set[str] = set()
        self._active_session_counts = [0] * config.num_workers
        self._in_flight_counts = [0] * config.num_workers
        self._routing_lock = threading.Lock()
        self._started = False
        self._fatal_error: GatewayError | None = None

    @property
    def num_workers(self) -> int:
        return self._config.num_workers

    def owner(self, session_id: str) -> int:
        with self._routing_lock:
            owner = self._session_owners.get(session_id)
        if owner is None:
            raise GatewayError(f"session {session_id!r} was not found", 404)
        return owner

    def start(self) -> None:
        if self._started:
            return
        self._response_queue = self._context.Queue()
        config_data = self._config.worker_config().model_dump(mode="json")
        for worker_id in range(self.num_workers):
            request_queue = self._context.Queue()
            process = self._context.Process(
                target=worker_main,
                args=(worker_id, config_data, request_queue, self._response_queue),
                name=f"rllm-gateway-worker-{worker_id}",
            )
            process.start()
            self._request_queues.append(request_queue)
            self._processes.append(process)
        self._started = True
        try:
            self._wait_until_ready()
        except Exception:
            self.stop()
            raise
        self._dispatcher = threading.Thread(target=self._dispatch_responses, name="gateway-response-dispatcher", daemon=True)
        self._dispatcher.start()

    async def call(self, session_id: str, operation: str, payload: dict[str, Any]) -> Any:
        return await self.call_worker(self.owner(session_id), operation, payload, session_id=session_id)

    async def create_session(self, session_id: str, payload: dict[str, Any]) -> None:
        with self._routing_lock:
            if session_id in self._session_owners:
                raise GatewayError(f"session {session_id!r} already exists", 409)
            worker_id = min(
                range(self.num_workers),
                key=lambda candidate: (self._active_session_counts[candidate], self._in_flight_counts[candidate], candidate),
            )
            self._session_owners[session_id] = worker_id
            self._active_sessions.add(session_id)
            self._active_session_counts[worker_id] += 1
        try:
            await self.call_worker(worker_id, "create_session", payload, session_id=session_id)
        except Exception:
            self._release_session(session_id)
            raise

    async def delete_session(self, session_id: str, payload: dict[str, Any]) -> None:
        worker_id = self.owner(session_id)
        self._release_session(session_id)
        self._fail_session_generations(session_id)
        await self.call_worker(worker_id, "delete_session", payload)

    async def call_worker(
        self,
        worker_id: int,
        operation: str,
        payload: dict[str, Any],
        session_id: str | None = None,
    ) -> Any:
        if not self._started:
            raise GatewayError("worker pool is not running", 503, "server_error")
        self._fail_dead_workers()
        if self._fatal_error is not None:
            raise self._fatal_error
        call_id = f"ipc_{uuid.uuid4().hex}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        with self._pending_lock:
            self._pending[call_id] = (worker_id, session_id, operation, loop, future)
        item = {"call_id": call_id, "operation": operation, "payload": payload}
        with self._routing_lock:
            self._in_flight_counts[worker_id] += 1
        self._request_queues[worker_id].put_nowait(item)
        try:
            return await asyncio.wait_for(future, timeout=self._config.request_timeout_seconds)
        except TimeoutError as exc:
            with self._pending_lock:
                self._pending.pop(call_id, None)
            raise GatewayError("gateway worker request timed out", 504, "timeout_error") from exc
        finally:
            with self._routing_lock:
                self._in_flight_counts[worker_id] -= 1

    def health(self) -> None:
        self._fail_dead_workers()
        if self._fatal_error is not None:
            raise self._fatal_error

    def stop(self) -> None:
        if not self._started:
            return
        for request_queue in self._request_queues:
            try:
                request_queue.put(None, timeout=1)
            except queue.Full:
                pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        self._response_queue.put(None)
        if self._dispatcher is not None:
            self._dispatcher.join(timeout=2)
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for _, _, _, loop, future in pending:
            loop.call_soon_threadsafe(_fail_future, future, GatewayError("worker pool stopped", 503, "server_error"))
        for request_queue in self._request_queues:
            request_queue.close()
        self._response_queue.close()
        self._request_queues.clear()
        self._processes.clear()
        with self._routing_lock:
            self._session_owners.clear()
            self._active_sessions.clear()
            self._active_session_counts = [0] * self.num_workers
            self._in_flight_counts = [0] * self.num_workers
        self._fatal_error = None
        self._started = False

    def _wait_until_ready(self) -> None:
        waiting = set(range(self.num_workers))
        deadline = time.monotonic() + self._config.request_timeout_seconds
        while waiting:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("gateway workers did not become ready before the startup timeout")
            try:
                result = self._response_queue.get(timeout=min(0.5, remaining))
            except queue.Empty:
                dead = {worker_id for worker_id in waiting if not self._processes[worker_id].is_alive()}
                if dead:
                    worker_id = min(dead)
                    raise RuntimeError(f"gateway worker {worker_id} exited during startup with code {self._processes[worker_id].exitcode}")
                continue
            if result.get("type") != "startup":
                continue
            worker_id = int(result["worker_id"])
            if not result.get("ok"):
                error = str(result.get("error", "unknown error"))
                logger.error("gateway worker %d failed during startup:\n%s", worker_id, error)
                raise RuntimeError(f"gateway worker {worker_id} failed to start; see the worker traceback above")
            waiting.discard(worker_id)

    def _dispatch_responses(self) -> None:
        while True:
            try:
                result = self._response_queue.get(timeout=0.5)
            except queue.Empty:
                self._fail_dead_workers()
                continue
            if result is None:
                return
            if result.get("type") == "fatal":
                self._fail_gateway(int(result["worker_id"]), str(result.get("error", "unknown error")))
                continue
            call_id = result.get("call_id")
            with self._pending_lock:
                pending = self._pending.pop(call_id, None)
            if pending is None:
                continue
            _, _, _, loop, future = pending
            if result.get("ok"):
                loop.call_soon_threadsafe(_resolve_future, future, result.get("value"))
            else:
                error = result.get("error") or {}
                exc = GatewayError(
                    str(error.get("message", "worker request failed")),
                    int(error.get("status_code", 500)),
                    str(error.get("error_type", "server_error")),
                )
                loop.call_soon_threadsafe(_fail_future, future, exc)

    def _fail_dead_workers(self) -> None:
        dead = {worker_id for worker_id, process in enumerate(self._processes) if not process.is_alive()}
        if not dead:
            return
        worker_id = min(dead)
        exit_code = self._processes[worker_id].exitcode
        if exit_code is not None and exit_code < 0:
            reason = f"terminated by signal {-exit_code}"
        else:
            reason = f"exited with code {exit_code}"
        self._fail_gateway(worker_id, reason)

    def _fail_gateway(self, worker_id: int, reason: str) -> None:
        logger.error("gateway worker %d failed: %s", worker_id, reason)
        if self._fatal_error is not None:
            return
        self._fatal_error = WorkerUnavailableError(worker_id)
        with self._pending_lock:
            failures = list(self._pending.values())
            self._pending.clear()
        for _, _, _, loop, future in failures:
            loop.call_soon_threadsafe(_fail_future, future, self._fatal_error)

    def _fail_session_generations(self, session_id: str) -> None:
        failures: list[tuple[int, str | None, str, asyncio.AbstractEventLoop, asyncio.Future[Any]]] = []
        with self._pending_lock:
            for call_id, pending in list(self._pending.items()):
                if pending[1] == session_id and pending[2] == "generate":
                    failures.append(pending)
                    self._pending.pop(call_id, None)
        for _, _, _, loop, future in failures:
            loop.call_soon_threadsafe(_fail_future, future, GatewayError("session was deleted", 410))

    def _release_session(self, session_id: str) -> None:
        with self._routing_lock:
            worker_id = self._session_owners.pop(session_id, None)
            if worker_id is None:
                return
            if session_id in self._active_sessions:
                self._active_sessions.remove(session_id)
                self._active_session_counts[worker_id] -= 1


def _resolve_future(future: asyncio.Future[Any], value: Any) -> None:
    if not future.done():
        future.set_result(value)


def _fail_future(future: asyncio.Future[Any], exc: Exception) -> None:
    if not future.done():
        future.set_exception(exc)

"""Training-side client for interacting with the model gateway."""

from typing import Any

import httpx

from rllm_model_gateway.models import TraceRecord, WorkerInfo


# Connection tuning shared by both clients.
#
# Keepalive stays disabled (max_keepalive_connections=0) so every request opens a fresh
# TCP connection. That avoids the race where the client and uvicorn both expire an idle
# connection at ~5s and the next request lands on a half-closed socket (httpx.ReadError).
#
# What is bounded here is *connect*. httpx's four phases are independent, and a scalar
# timeout sets all of them: the trace API passes 600s, so a stalled connect hung the
# rollout for ten minutes with the gateway completely idle -- which reads as "the run is
# stuck on its last few trajectories" rather than as a connection problem.
#
# Only connect is capped. `read` keeps the caller's full budget, so a slow *response* is
# unaffected; and generations never come through here anyway -- they use Proxy's own
# client, which deliberately runs with no timeout at all.
#
# 30s, not something tighter: a connect to the gateway is same-node, so under a
# millisecond, and the pathological case stalled indefinitely -- any finite bound
# separates them. The cost of being wrong is real, because a ConnectTimeout propagates
# out of aget_traces and makes process_task_with_retry re-run the whole rollout
# (generation included), up to retry_limit; with the default raise_on_error=True a
# persistent failure then takes the run down. So leave headroom for a transient stall and
# still turn a 30-minute hang into a ~90s diagnosable failure.
_CONNECT_TIMEOUT_S = 30.0
# Inert as things stand: _limits() leaves max_connections unbounded, so httpx never waits
# for a pool slot. Kept as a guard in case that cap is ever introduced.
_POOL_TIMEOUT_S = 60.0


def _limits() -> "httpx.Limits":
    return httpx.Limits(max_keepalive_connections=0)


def _timeout(timeout: float) -> "httpx.Timeout":
    """Caller's value governs read/write; connect and pool are bounded separately."""
    return httpx.Timeout(timeout, connect=_CONNECT_TIMEOUT_S, pool=_POOL_TIMEOUT_S)


class GatewayClient:
    """Synchronous client for the rllm-model-gateway REST API.

    Intended for use by the training framework to create sessions, retrieve
    traces, and manage workers.
    """

    def __init__(
        self,
        gateway_url: str,
        timeout: float = 30.0,
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        self._http = httpx.Client(timeout=_timeout(timeout), limits=_limits())

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- Session lifecycle -------------------------------------------------

    def create_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> str:
        """Create a session (or let the gateway generate an ID)."""
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        if metadata:
            body["metadata"] = metadata
        if sampling_params:
            body["sampling_params"] = sampling_params
        resp = self._http.post(f"{self.gateway_url}/sessions", json=body)
        resp.raise_for_status()
        return resp.json()["session_id"]

    def get_session_url(self, session_id: str) -> str:
        """Return the OpenAI-compatible base URL for an agent to use."""
        return f"{self.gateway_url}/sessions/{session_id}/v1"

    def get_session_info(self, session_id: str) -> dict[str, Any]:
        resp = self._http.get(f"{self.gateway_url}/sessions/{session_id}")
        resp.raise_for_status()
        return resp.json()

    def list_sessions(self, since: float | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        resp = self._http.get(f"{self.gateway_url}/sessions", params=params)
        resp.raise_for_status()
        return resp.json()

    def delete_session(self, session_id: str) -> int:
        resp = self._http.delete(f"{self.gateway_url}/sessions/{session_id}")
        resp.raise_for_status()
        return resp.json().get("deleted", 0)

    # -- Trace retrieval ---------------------------------------------------

    def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[TraceRecord]:
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        resp = self._http.get(f"{self.gateway_url}/sessions/{session_id}/traces", params=params)
        resp.raise_for_status()
        data = resp.json()
        return [TraceRecord(**t) for t in data]

    def get_trace(self, trace_id: str) -> TraceRecord:
        resp = self._http.get(f"{self.gateway_url}/traces/{trace_id}")
        resp.raise_for_status()
        return TraceRecord(**resp.json())

    # -- Worker management -------------------------------------------------

    def add_worker(
        self,
        url: str,
        api_path: str = "/v1",
        model_name: str | None = None,
        weight: int = 1,
    ) -> str:
        """Register a worker.  Returns worker_id."""
        body: dict[str, Any] = {"url": url, "api_path": api_path, "weight": weight}
        if model_name:
            body["model_name"] = model_name
        resp = self._http.post(f"{self.gateway_url}/admin/workers", json=body)
        resp.raise_for_status()
        return resp.json()["worker_id"]

    def remove_worker(self, worker_id: str) -> None:
        resp = self._http.delete(f"{self.gateway_url}/admin/workers/{worker_id}")
        resp.raise_for_status()

    def list_workers(self) -> list[WorkerInfo]:
        resp = self._http.get(f"{self.gateway_url}/admin/workers")
        resp.raise_for_status()
        return [WorkerInfo(**w) for w in resp.json()]

    # -- Weight version ----------------------------------------------------

    def set_weight_version(self, weight_version: int) -> int:
        resp = self._http.post(f"{self.gateway_url}/admin/weight_version", json={"weight_version": weight_version})
        resp.raise_for_status()
        return resp.json()["weight_version"]

    def get_weight_version(self) -> int | None:
        resp = self._http.get(f"{self.gateway_url}/admin/weight_version")
        resp.raise_for_status()
        return resp.json().get("weight_version")

    # -- Lifecycle ---------------------------------------------------------

    def flush(self, timeout: float = 30.0) -> bool:
        # _timeout(), not the bare float: a per-request *scalar* replaces the client
        # default outright, so passing `timeout` here would restore connect=timeout and
        # undo the cap set above. flush is on the per-rollout path, so that matters:
        # aget_traces calls it before reading traces, once per rollout.
        resp = self._http.post(f"{self.gateway_url}/admin/flush", timeout=_timeout(timeout))
        resp.raise_for_status()
        return resp.json().get("status") == "flushed"

    def health(self) -> dict[str, Any]:
        resp = self._http.get(f"{self.gateway_url}/health")
        resp.raise_for_status()
        return resp.json()


class AsyncGatewayClient:
    """Async variant of :class:`GatewayClient` using ``httpx.AsyncClient``."""

    def __init__(
        self,
        gateway_url: str,
        timeout: float = 30.0,
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=_timeout(timeout), limits=_limits())

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    # -- Session lifecycle -------------------------------------------------

    async def create_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        if metadata:
            body["metadata"] = metadata
        if sampling_params:
            body["sampling_params"] = sampling_params
        resp = await self._http.post(f"{self.gateway_url}/sessions", json=body)
        resp.raise_for_status()
        return resp.json()["session_id"]

    def get_session_url(self, session_id: str) -> str:
        return f"{self.gateway_url}/sessions/{session_id}/v1"

    async def get_session_info(self, session_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{self.gateway_url}/sessions/{session_id}")
        resp.raise_for_status()
        return resp.json()

    async def list_sessions(self, since: float | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        resp = await self._http.get(f"{self.gateway_url}/sessions", params=params)
        resp.raise_for_status()
        return resp.json()

    async def delete_session(self, session_id: str) -> int:
        resp = await self._http.delete(f"{self.gateway_url}/sessions/{session_id}")
        resp.raise_for_status()
        return resp.json().get("deleted", 0)

    async def delete_sessions(self, session_ids: list[str]) -> int:
        """Batch-delete sessions (and their traces) in a single round-trip."""
        if not session_ids:
            return 0
        resp = await self._http.post(
            f"{self.gateway_url}/sessions/batch_delete",
            json={"session_ids": list(session_ids)},
        )
        resp.raise_for_status()
        return resp.json().get("deleted", 0)

    # -- Trace retrieval ---------------------------------------------------

    async def get_session_traces(
        self,
        session_id: str,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[TraceRecord]:
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        resp = await self._http.get(f"{self.gateway_url}/sessions/{session_id}/traces", params=params)
        resp.raise_for_status()
        data = resp.json()
        return [TraceRecord(**t) for t in data]

    async def get_trace(self, trace_id: str) -> TraceRecord:
        resp = await self._http.get(f"{self.gateway_url}/traces/{trace_id}")
        resp.raise_for_status()
        return TraceRecord(**resp.json())

    # -- Worker management -------------------------------------------------

    async def add_worker(
        self,
        url: str,
        api_path: str = "/v1",
        model_name: str | None = None,
        weight: int = 1,
    ) -> str:
        body: dict[str, Any] = {"url": url, "api_path": api_path, "weight": weight}
        if model_name:
            body["model_name"] = model_name
        resp = await self._http.post(f"{self.gateway_url}/admin/workers", json=body)
        resp.raise_for_status()
        return resp.json()["worker_id"]

    async def remove_worker(self, worker_id: str) -> None:
        resp = await self._http.delete(f"{self.gateway_url}/admin/workers/{worker_id}")
        resp.raise_for_status()

    async def list_workers(self) -> list[WorkerInfo]:
        resp = await self._http.get(f"{self.gateway_url}/admin/workers")
        resp.raise_for_status()
        return [WorkerInfo(**w) for w in resp.json()]

    # -- Weight version ----------------------------------------------------

    async def set_weight_version(self, weight_version: int) -> int:
        resp = await self._http.post(f"{self.gateway_url}/admin/weight_version", json={"weight_version": weight_version})
        resp.raise_for_status()
        return resp.json()["weight_version"]

    async def get_weight_version(self) -> int | None:
        resp = await self._http.get(f"{self.gateway_url}/admin/weight_version")
        resp.raise_for_status()
        return resp.json().get("weight_version")

    # -- Lifecycle ---------------------------------------------------------

    async def flush(self, timeout: float = 30.0) -> bool:
        # See GatewayClient.flush: a scalar here would undo the connect bound.
        resp = await self._http.post(f"{self.gateway_url}/admin/flush", timeout=_timeout(timeout))
        resp.raise_for_status()
        return resp.json().get("status") == "flushed"

    async def health(self) -> dict[str, Any]:
        resp = await self._http.get(f"{self.gateway_url}/health")
        resp.raise_for_status()
        return resp.json()

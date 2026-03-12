"""Training-side client for interacting with the model gateway."""

from typing import Any

import httpx

from rllm_model_gateway.models import TraceRecord, WorkerInfo


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
        self._http = httpx.Client(timeout=timeout)

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
    ) -> str:
        """Create a session (or let the gateway generate an ID)."""
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        if metadata:
            body["metadata"] = metadata
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
        model_name: str | None = None,
        weight: int = 1,
    ) -> str:
        """Register a worker.  Returns worker_id."""
        body: dict[str, Any] = {"url": url, "weight": weight}
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

    # -- Lifecycle ---------------------------------------------------------

    def flush(self, timeout: float = 30.0) -> bool:
        resp = self._http.post(f"{self.gateway_url}/admin/flush", timeout=timeout)
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
        self._http = httpx.AsyncClient(timeout=timeout)

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
    ) -> str:
        body: dict[str, Any] = {}
        if session_id:
            body["session_id"] = session_id
        if metadata:
            body["metadata"] = metadata
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
        model_name: str | None = None,
        weight: int = 1,
    ) -> str:
        body: dict[str, Any] = {"url": url, "weight": weight}
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

    # -- Lifecycle ---------------------------------------------------------

    async def flush(self, timeout: float = 30.0) -> bool:
        resp = await self._http.post(f"{self.gateway_url}/admin/flush", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("status") == "flushed"

    async def health(self) -> dict[str, Any]:
        resp = await self._http.get(f"{self.gateway_url}/health")
        resp.raise_for_status()
        return resp.json()

"""REST clients for the gateway API."""

from __future__ import annotations

from typing import Any

import httpx

from rllm_model_gateway.trace import TraceRecord


class GatewayClient:
    def __init__(
        self,
        gateway_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Sync client.

        Pass the gateway's admin_api_key to access management endpoints
        (sessions, traces, admin). Reading config from a freshly created
        gateway exposes the key as ``app.state.config.admin_api_key``.
        """
        self.gateway_url = gateway_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(timeout=timeout, headers=headers)

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- Sessions ------------------------------------------------------

    def create_session(
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
        resp = self._http.post(f"{self.gateway_url}/sessions", json=body)
        resp.raise_for_status()
        return resp.json()["session_id"]

    def get_session_url(self, session_id: str) -> str:
        """Base URL for OpenAI-style SDKs (includes ``/v1`` suffix)."""
        return f"{self.gateway_url}/sessions/{session_id}/v1"

    def get_anthropic_session_url(self, session_id: str) -> str:
        """Base URL for the Anthropic Python SDK (no ``/v1`` suffix; SDK
        appends its own version-prefixed path)."""
        return f"{self.gateway_url}/sessions/{session_id}"

    def get_session(self, session_id: str) -> dict[str, Any]:
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

    # -- Traces --------------------------------------------------------

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
        return [TraceRecord(**t) for t in resp.json()]

    def get_trace(self, trace_id: str) -> TraceRecord:
        resp = self._http.get(f"{self.gateway_url}/traces/{trace_id}")
        resp.raise_for_status()
        return TraceRecord(**resp.json())

    def get_trace_extras(self, trace_id: str) -> tuple[str, bytes] | None:
        resp = self._http.get(f"{self.gateway_url}/traces/{trace_id}/extras")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        fmt = resp.headers.get("X-Extras-Format", "msgpack")
        return (fmt, resp.content)

    # -- Lifecycle -----------------------------------------------------

    def flush(self, timeout: float = 30.0) -> bool:
        resp = self._http.post(f"{self.gateway_url}/admin/flush", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("status") == "flushed"

    def health(self) -> dict[str, Any]:
        resp = self._http.get(f"{self.gateway_url}/health")
        resp.raise_for_status()
        return resp.json()


class AsyncGatewayClient:
    def __init__(
        self,
        gateway_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    # -- Sessions ------------------------------------------------------

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

    def get_anthropic_session_url(self, session_id: str) -> str:
        return f"{self.gateway_url}/sessions/{session_id}"

    async def get_session(self, session_id: str) -> dict[str, Any]:
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

    # -- Traces --------------------------------------------------------

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
        return [TraceRecord(**t) for t in resp.json()]

    async def get_trace(self, trace_id: str) -> TraceRecord:
        resp = await self._http.get(f"{self.gateway_url}/traces/{trace_id}")
        resp.raise_for_status()
        return TraceRecord(**resp.json())

    async def get_trace_extras(self, trace_id: str) -> tuple[str, bytes] | None:
        resp = await self._http.get(f"{self.gateway_url}/traces/{trace_id}/extras")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        fmt = resp.headers.get("X-Extras-Format", "msgpack")
        return (fmt, resp.content)

    # -- Lifecycle -----------------------------------------------------

    async def flush(self, timeout: float = 30.0) -> bool:
        resp = await self._http.post(f"{self.gateway_url}/admin/flush", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("status") == "flushed"

    async def health(self) -> dict[str, Any]:
        resp = await self._http.get(f"{self.gateway_url}/health")
        resp.raise_for_status()
        return resp.json()

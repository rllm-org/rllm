from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel

from rllm_model_gateway_v2.contracts import SessionTraces


class SessionCredentials(BaseModel):
    session_id: str
    url: str
    agent_key: str


class GatewayClient:
    def __init__(self, base_url: str, admin_key: str, timeout: float = 30.0) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=timeout,
        )

    def create_session(
        self,
        session_id: str | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> SessionCredentials:
        body = _session_body(session_id, sampling_params)
        response = self._client.post("/admin/sessions", json=body)
        response.raise_for_status()
        return SessionCredentials.model_validate(response.json())

    def get_session(self, session_id: str) -> SessionTraces:
        response = self._client.get(f"/admin/sessions/{_path(session_id)}")
        response.raise_for_status()
        return SessionTraces.model_validate(response.json())

    def delete_session(self, session_id: str) -> None:
        response = self._client.delete(f"/admin/sessions/{_path(session_id)}")
        response.raise_for_status()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GatewayClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class AsyncGatewayClient:
    def __init__(self, base_url: str, admin_key: str, timeout: float = 30.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=timeout,
        )

    async def create_session(
        self,
        session_id: str | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> SessionCredentials:
        body = _session_body(session_id, sampling_params)
        response = await self._client.post("/admin/sessions", json=body)
        response.raise_for_status()
        return SessionCredentials.model_validate(response.json())

    async def get_session(self, session_id: str) -> SessionTraces:
        response = await self._client.get(f"/admin/sessions/{_path(session_id)}")
        response.raise_for_status()
        return SessionTraces.model_validate(response.json())

    async def delete_session(self, session_id: str) -> None:
        response = await self._client.delete(f"/admin/sessions/{_path(session_id)}")
        response.raise_for_status()

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncGatewayClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()


def _path(session_id: str) -> str:
    return quote(session_id, safe="/")


def _session_body(
    session_id: str | None,
    sampling_params: dict[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"sampling_params": sampling_params or {}}
    if session_id is not None:
        body["session_id"] = session_id
    return body

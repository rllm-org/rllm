from typing import Any

import pytest
import rllm_model_gateway.v2.server as server_module
from fastapi.testclient import TestClient
from rllm_model_gateway.v2.config import GatewayConfig
from rllm_model_gateway.v2.types import GatewayError, GatewayResponse


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def poll(self, timeout: float) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


class FakeWorkerPool:
    instances: list["FakeWorkerPool"] = []

    def __init__(self, config, inference_client_cls, inference_client_kwargs, shutdown) -> None:
        self.started = False
        self.stopped = False
        self.sessions: dict[str, dict[str, Any]] = {}
        self.generate_requests: list[dict[str, Any]] = []
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def health(self) -> None:
        pass

    async def create_session(self, session_id: str, payload: dict[str, Any]) -> None:
        if session_id in self.sessions:
            raise GatewayError("duplicate session", 409, "conflict_error")
        self.sessions[session_id] = payload

    async def call(self, session_id: str, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if session_id not in self.sessions:
            raise GatewayError("session not found", 404, "not_found_error")
        if operation == "get_session_traces":
            return {"session_id": session_id, "traces": [], "created_at": 1.0}
        if operation == "generate":
            request = payload["request"]
            self.generate_requests.append(request)
            return GatewayResponse(
                request_id=request["request_id"],
                text="decoded response",
                content="response",
                reasoning_content="reasoning",
                tool_calls=[],
                finish_reason="stop",
                prompt_tokens=3,
                completion_tokens=2,
            ).model_dump(mode="json")
        raise AssertionError(f"unexpected operation: {operation}")

    async def delete_session(self, session_id: str, payload: dict[str, Any]) -> None:
        if self.sessions.pop(session_id, None) is None:
            raise GatewayError("session not found", 404, "not_found_error")

    async def update_inference_client(self, update: dict[str, Any]) -> None:
        pass


class UnusedInferenceClient:
    pass


@pytest.fixture
def v2_server(monkeypatch):
    FakeWorkerPool.instances.clear()
    monkeypatch.setattr(server_module, "WorkerPool", FakeWorkerPool)
    connection = FakeConnection()
    app = server_module.create_app(
        GatewayConfig(admin_key="admin-key", tokenizer_model="unused"),
        UnusedInferenceClient,
        {},
        connection,
        lambda: None,
    )
    with TestClient(app) as client:
        yield client, FakeWorkerPool.instances[0]
    assert connection.closed


def test_admin_and_agent_lifecycle(v2_server) -> None:
    client, pool = v2_server

    unauthorized = client.post("/admin/sessions", json={"session_id": "session-1"})
    assert unauthorized.status_code == 401

    created = client.post(
        "/admin/sessions",
        headers={"Authorization": "Bearer admin-key"},
        json={"session_id": "session-1", "sampling_params": {"temperature": 0.4}},
    )
    assert created.status_code == 200
    agent_key = created.json()["agent_key"]
    assert pool.sessions["session-1"]["sampling_params"] == {"temperature": 0.4}

    wrong_authority = client.post(
        "/sessions/session-1/v1/chat/completions",
        headers={"Authorization": "Bearer admin-key"},
        json={"model": "model", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert wrong_authority.status_code == 401

    generated = client.post(
        "/sessions/session-1/v1/chat/completions",
        headers={"Authorization": f"Bearer {agent_key}"},
        json={
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
        },
    )
    assert generated.status_code == 200
    assert generated.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "response",
        "reasoning_content": "reasoning",
    }
    assert pool.generate_requests[0]["sampling_params"] == {"temperature": 0.7}

    traces = client.get(
        "/admin/sessions/session-1",
        headers={"Authorization": "Bearer admin-key"},
    )
    assert traces.status_code == 200
    assert traces.json()["session_id"] == "session-1"

    deleted = client.delete(
        "/admin/sessions/session-1",
        headers={"Authorization": "Bearer admin-key"},
    )
    assert deleted.status_code == 204

    revoked = client.post(
        "/sessions/session-1/v1/chat/completions",
        headers={"Authorization": f"Bearer {agent_key}"},
        json={"model": "model", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert revoked.status_code == 401


def test_completion_token_ids_and_synthetic_streaming(v2_server) -> None:
    client, pool = v2_server
    created = client.post(
        "/admin/sessions",
        headers={"Authorization": "Bearer admin-key"},
        json={"session_id": "session-1"},
    )
    agent_key = created.json()["agent_key"]

    response = client.post(
        "/sessions/session-1/v1/completions",
        headers={"Authorization": f"Bearer {agent_key}"},
        json={"model": "model", "prompt": [1, 2, 3], "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text
    assert pool.generate_requests[0]["prompt_token_ids"] == [1, 2, 3]
    assert pool.generate_requests[0]["sampling_params"] == {}


def test_stream_options_require_streaming(v2_server) -> None:
    client, _ = v2_server
    created = client.post(
        "/admin/sessions",
        headers={"Authorization": "Bearer admin-key"},
        json={"session_id": "session-1"},
    )
    agent_key = created.json()["agent_key"]

    response = client.post(
        "/sessions/session-1/v1/chat/completions",
        headers={"Authorization": f"Bearer {agent_key}"},
        json={
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "stream_options may only be used when stream is true"

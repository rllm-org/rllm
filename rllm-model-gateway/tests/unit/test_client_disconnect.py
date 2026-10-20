"""A non-streaming call whose client disconnects is cancelled at the worker, and the gateway answers 499."""

import asyncio
import threading

import httpx
import openai
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from rllm_model_gateway import GatewayClient, GatewayConfig, create_app

from tests.helpers.gateway_server import GatewayServer
from tests.helpers.mock_vllm import MOCK_RESPONSE
from tests.unit.test_cumulative_token_mode import _MockRenderer


class HangingWorker:
    """A worker whose completions never answer, recording when the gateway closes the request.

    With `answer_chat`, its chat completions answer with the mock vLLM reply instead, so a
    cumulative session's first turn completes and its second, sent to /v1/completions, hangs.
    """

    def __init__(self, answer_chat: bool = False) -> None:
        self.arrived = threading.Event()
        self.closed = threading.Event()
        app = FastAPI()

        async def hang(request: Request):
            self.arrived.set()
            while not await request.is_disconnected():
                await asyncio.sleep(0.01)
            self.closed.set()

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            if answer_chat:
                return JSONResponse(content=MOCK_RESPONSE)
            return await hang(request)

        @app.post("/v1/completions")
        async def completions(request: Request):
            return await hang(request)

        self.server = GatewayServer(app)


class Statuses:
    """The gateway app, recording the status of each response it starts."""

    def __init__(self, app) -> None:
        self.app = app
        self.seen: list[int] = []
        self.answered = threading.Event()

    async def __call__(self, scope, receive, send):
        async def sending(message):
            if message["type"] == "http.response.start":
                self.seen.append(message["status"])
                self.answered.set()
            await send(message)

        await self.app(scope, receive, sending)


def start(worker: HangingWorker, cumulative: bool = False) -> tuple[GatewayServer, Statuses]:
    worker.server.start()
    config = GatewayConfig(store_worker="memory", workers=[{"url": f"{worker.server.url}/v1", "worker_id": "w0"}], health_check_interval=999, sync_traces=True)
    app = create_app(config)
    if cumulative:
        app.state.proxy.renderer = _MockRenderer()
        app.state.proxy.cumulative_token_mode = True
    statuses = Statuses(app)
    gateway = GatewayServer(statuses)
    gateway.start()
    return gateway, statuses


@pytest.fixture
def servers():
    started = []
    yield started
    for server in reversed(started):
        server.stop()


def test_a_client_that_disconnects_cancels_its_call_at_the_worker(servers):
    """The client gives up while the worker is still generating; the gateway closes the
    worker's request rather than waiting on it, answers 499, and records no trace."""
    worker = HangingWorker()
    gateway, statuses = start(worker)
    servers.extend([worker.server, gateway])
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(f"{gateway.url}/sessions/gone/v1/chat/completions", timeout=1.0, json={"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]})
    assert worker.arrived.is_set()
    assert worker.closed.wait(10.0)
    assert statuses.answered.wait(10.0)
    assert statuses.seen == [499]
    assert GatewayClient(gateway.url).get_session_traces("gone") == []


def test_a_cumulative_turn_whose_client_disconnects_cancels_its_call_at_the_worker(servers):
    """In cumulative token mode a later turn goes to /v1/completions; a client that gives up on it
    closes that request at the worker too, and the gateway answers it 499 and records no trace."""
    worker = HangingWorker(answer_chat=True)
    gateway, statuses = start(worker, cumulative=True)
    servers.extend([worker.server, gateway])
    oai = openai.OpenAI(base_url=f"{gateway.url}/sessions/gone-later/v1", api_key="dummy")
    oai.chat.completions.create(model="mock-model", messages=[{"role": "user", "content": "Hello"}])
    assert statuses.seen == [200]
    statuses.answered.clear()
    later = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello from mock!"}, {"role": "user", "content": "How are you?"}]
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(f"{gateway.url}/sessions/gone-later/v1/chat/completions", timeout=1.0, json={"model": "mock-model", "messages": later})
    assert worker.arrived.is_set()
    assert worker.closed.wait(10.0)
    assert statuses.answered.wait(10.0)
    assert statuses.seen == [200, 499]
    assert len(GatewayClient(gateway.url).get_session_traces("gone-later")) == 1

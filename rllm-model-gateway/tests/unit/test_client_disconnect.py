"""A non-streaming call whose client disconnects is cancelled at the worker, and the gateway answers 499."""

import asyncio
import contextlib
import threading

import httpx
import openai
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from rllm_model_gateway import GatewayClient, GatewayConfig, create_app

from tests.helpers.gateway_server import GatewayServer
from tests.helpers.mock_renderer import MockRenderer
from tests.helpers.mock_vllm import MOCK_RESPONSE


class HangingWorker:
    """A worker whose completions answer only once their request's connection closes, or once the
    test releases them, recording when the gateway closes the request.

    With `answer_chat`, its chat completions answer with the mock vLLM reply instead, so a
    cumulative session's first turn completes and its second, sent to /v1/completions, hangs.
    """

    def __init__(self, answer_chat: bool = False) -> None:
        self.arrived = threading.Event()
        self.closed = threading.Event()
        self.released = threading.Event()
        app = FastAPI()

        async def hang(request: Request):
            self.arrived.set()
            while not self.released.is_set():
                if await request.is_disconnected():
                    self.closed.set()
                    return
                await asyncio.sleep(0.01)
            return JSONResponse(content=MOCK_RESPONSE)

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


@pytest.fixture
def started():
    """Starts a worker and a gateway over it; at teardown, releases the worker's pending requests
    before stopping both, so a gateway still waiting on one can shut down."""
    running = []

    def start(worker: HangingWorker, cumulative: bool = False) -> tuple[GatewayServer, Statuses]:
        worker.server.start()
        running.append(worker)
        config = GatewayConfig(store_worker="memory", workers=[{"url": f"{worker.server.url}/v1", "worker_id": "w0"}], health_check_interval=999, sync_traces=True)
        app = create_app(config)
        if cumulative:
            app.state.proxy.renderer = MockRenderer()
            app.state.proxy.cumulative_token_mode = True
        statuses = Statuses(app)
        gateway = GatewayServer(statuses)
        gateway.start()
        running.append(gateway)
        return gateway, statuses

    yield start
    for held in running:
        if isinstance(held, HangingWorker):
            held.released.set()
    for held in reversed(running):
        (held.server if isinstance(held, HangingWorker) else held).stop()


def disconnect_once_arrived(url: str, messages: list[dict], worker: HangingWorker) -> None:
    """Sends a chat completion, waits until the worker holds it, then closes the client's connection."""

    async def run():
        async with httpx.AsyncClient(timeout=None) as client:
            request = asyncio.ensure_future(client.post(url, json={"model": "mock-model", "messages": messages}))
            arrived = await asyncio.to_thread(worker.arrived.wait, 10.0)
            request.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await request
        assert arrived

    asyncio.run(run())


def test_a_client_that_disconnects_cancels_its_call_at_the_worker(started):
    """The client gives up while the worker is still generating; the gateway closes the
    worker's request rather than waiting on it, answers 499, and records no trace."""
    worker = HangingWorker()
    gateway, statuses = started(worker)
    disconnect_once_arrived(f"{gateway.url}/sessions/gone/v1/chat/completions", [{"role": "user", "content": "hi"}], worker)
    assert worker.closed.wait(10.0)
    assert statuses.answered.wait(10.0)
    assert statuses.seen == [499]
    assert GatewayClient(gateway.url).get_session_traces("gone") == []


def test_a_cumulative_turn_whose_client_disconnects_cancels_its_call_at_the_worker(started):
    """In cumulative token mode a later turn goes to /v1/completions; a client that gives up on it
    closes that request at the worker too, and the gateway answers it 499 and records no trace."""
    worker = HangingWorker(answer_chat=True)
    gateway, statuses = started(worker, cumulative=True)
    oai = openai.OpenAI(base_url=f"{gateway.url}/sessions/gone-later/v1", api_key="dummy")
    oai.chat.completions.create(model="mock-model", messages=[{"role": "user", "content": "Hello"}])
    assert statuses.seen == [200]
    statuses.answered.clear()
    later = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello from mock!"}, {"role": "user", "content": "How are you?"}]
    disconnect_once_arrived(f"{gateway.url}/sessions/gone-later/v1/chat/completions", later, worker)
    assert worker.closed.wait(10.0)
    assert statuses.answered.wait(10.0)
    assert statuses.seen == [200, 499]
    assert len(GatewayClient(gateway.url).get_session_traces("gone-later")) == 1

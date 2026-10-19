"""A non-streaming call whose client disconnects is cancelled at the worker."""

import asyncio
import threading

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from rllm_model_gateway import GatewayConfig, create_app

from tests.helpers.gateway_server import GatewayServer, _reserve_local_port


class HangingWorker:
    """A worker whose completions never answer; it records when the gateway closes the request."""

    def __init__(self) -> None:
        self.arrived = threading.Event()
        self.closed = threading.Event()
        app = FastAPI()

        @app.post("/v1/chat/completions")
        @app.post("/v1/completions")
        async def completions(request: Request):
            self.arrived.set()
            while not await request.is_disconnected():
                await asyncio.sleep(0.01)
            self.closed.set()

        self.host, self.port = "127.0.0.1", _reserve_local_port("127.0.0.1")
        self._server = uvicorn.Server(uvicorn.Config(app, host=self.host, port=self.port, log_level="error"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self._thread.start()
        while not self._server.started:
            self._thread.join(0.05)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(5.0)


@pytest.fixture
def worker():
    server = HangingWorker()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def gateway(worker):
    config = GatewayConfig(store_worker="memory", workers=[{"url": f"{worker.url}/v1", "worker_id": "w0"}], health_check_interval=999, sync_traces=True)
    server = GatewayServer(create_app(config))
    server.start()
    yield server
    server.stop()


def test_a_client_that_disconnects_cancels_its_call_at_the_worker(worker, gateway):
    """The client gives up while the worker is still generating; the gateway closes the
    worker's request rather than waiting on it."""
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(f"{gateway.url}/sessions/gone/v1/chat/completions", timeout=1.0, json={"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]})
    assert worker.arrived.is_set()
    assert worker.closed.wait(10.0)

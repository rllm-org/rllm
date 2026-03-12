"""Shared mock vLLM server implementations for tests.

Provides:
- MockVLLMServer: basic mock with canned responses
- ControllableMockVLLMServer: adds admin endpoints for failure injection
"""

import json
import threading
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ------------------------------------------------------------------
# Canned response data
# ------------------------------------------------------------------

MOCK_RESPONSE = {
    "id": "chatcmpl-mock",
    "object": "chat.completion",
    "model": "mock-model",
    "prompt_token_ids": [1, 2, 3, 4, 5],
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from mock!"},
            "finish_reason": "stop",
            "stop_reason": None,
            "token_ids": [10, 11, 12],
            "logprobs": {
                "content": [
                    {"token": "Hello", "logprob": -0.5, "bytes": None, "top_logprobs": []},
                    {"token": " from", "logprob": -0.3, "bytes": None, "top_logprobs": []},
                    {"token": " mock!", "logprob": -0.1, "bytes": None, "top_logprobs": []},
                ]
            },
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    "prompt_logprobs": None,
    "kv_transfer_params": None,
}


def stream_chunks():
    """Yield SSE chunks that mirror vLLM 0.11+ streaming format."""
    chunks = [
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "mock-model",
            "prompt_token_ids": [1, 2, 3, 4, 5],
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "logprobs": {"content": [{"token": "Hello", "logprob": -0.5, "bytes": None, "top_logprobs": []}]},
                    "finish_reason": None,
                    "token_ids": [10],
                }
            ],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " from mock!"},
                    "logprobs": {
                        "content": [
                            {"token": " from", "logprob": -0.3, "bytes": None, "top_logprobs": []},
                            {"token": " mock!", "logprob": -0.1, "bytes": None, "top_logprobs": []},
                        ]
                    },
                    "finish_reason": None,
                    "token_ids": [11, 12],
                }
            ],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "mock-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "stop_reason": None}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        },
    ]
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


def build_mock_vllm_app() -> FastAPI:
    """Create a minimal mock vLLM server that returns canned responses."""
    app = FastAPI()
    app.state.request_log: list[dict[str, Any]] = []
    app.state._log_lock = threading.Lock()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "mock-model", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        with app.state._log_lock:
            app.state.request_log.append(body)

        if body.get("stream"):
            return StreamingResponse(stream_chunks(), media_type="text/event-stream")
        return JSONResponse(content=MOCK_RESPONSE)

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        with app.state._log_lock:
            app.state.request_log.append(body)
        return JSONResponse(content=MOCK_RESPONSE)

    return app


def build_controllable_mock_vllm_app(response_delay: float = 0.0) -> FastAPI:
    """Create a mock vLLM server with admin endpoints for failure injection.

    Admin endpoints:
    - POST /admin/set_fail    {"fail": true/false}  - toggle 500 errors
    - POST /admin/set_malformed {"malformed": true/false} - toggle invalid JSON
    """
    app = FastAPI()
    app.state.request_log: list[dict[str, Any]] = []
    app.state._log_lock = threading.Lock()
    app.state.should_fail = False
    app.state.should_malform = False
    app.state.response_delay = response_delay

    @app.get("/health")
    async def health():
        # Health endpoint always returns ok so the worker stays routable.
        # Only /v1/chat/completions is affected by should_fail.
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "mock-model", "object": "model"}]}

    @app.post("/admin/set_fail")
    async def set_fail(request: Request):
        body = await request.json()
        app.state.should_fail = body.get("fail", False)
        return {"should_fail": app.state.should_fail}

    @app.post("/admin/set_malformed")
    async def set_malformed(request: Request):
        body = await request.json()
        app.state.should_malform = body.get("malformed", False)
        return {"should_malform": app.state.should_malform}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        with app.state._log_lock:
            app.state.request_log.append(body)

        if app.state.response_delay > 0:
            import asyncio

            await asyncio.sleep(app.state.response_delay)

        if app.state.should_fail:
            return JSONResponse(status_code=500, content={"error": "simulated backend failure"})

        if app.state.should_malform:
            from starlette.responses import Response

            return Response(content="this is not valid json{{{", media_type="application/json")

        if body.get("stream"):
            return StreamingResponse(stream_chunks(), media_type="text/event-stream")
        return JSONResponse(content=MOCK_RESPONSE)

    return app


class MockVLLMServer:
    """Run a mock vLLM server in a background thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port
        self.app = build_mock_vllm_app()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def request_log(self) -> list[dict[str, Any]]:
        return self.app.state.request_log

    def start(self) -> None:
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="error",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._server.started:
                for sock in self._server.servers:
                    self.port = sock.sockets[0].getsockname()[1]
                return
            time.sleep(0.05)
        raise RuntimeError("Mock vLLM server failed to start")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)


class ControllableMockVLLMServer:
    """Mock vLLM server with admin endpoints for failure injection."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, response_delay: float = 0.0) -> None:
        self.host = host
        self.port = port
        self.app = build_controllable_mock_vllm_app(response_delay=response_delay)
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def request_log(self) -> list[dict[str, Any]]:
        return self.app.state.request_log

    def set_fail(self, fail: bool = True) -> None:
        self.app.state.should_fail = fail

    def set_malform(self, malform: bool = True) -> None:
        self.app.state.should_malform = malform

    def start(self) -> None:
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="error",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._server.started:
                for sock in self._server.servers:
                    self.port = sock.sockets[0].getsockname()[1]
                return
            time.sleep(0.05)
        raise RuntimeError("Controllable mock vLLM server failed to start")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)

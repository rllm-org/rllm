"""Mock SGLang server for use_sglang-mode tests.

SGLang's native generation API is ``POST /generate``: token-in (``input_ids``),
token-out via ``meta_info.output_token_logprobs`` — a list of
``[logprob, token_id, ...]`` rows (the canonical token-id + logprob source the
gateway's use_sglang path reads).
"""

import json
import socket
import threading
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Canned completion: 3 output tokens [10, 11, 12] with logprobs.
_OUTPUT_TOKEN_LOGPROBS = [[-0.5, 10, None], [-0.3, 11, None], [-0.1, 12, None]]
_OUTPUT_TEXT = "Hello from mock!"


def _generate_payload(input_ids: list[int]) -> dict[str, Any]:
    return {
        "text": _OUTPUT_TEXT,
        "meta_info": {
            "output_token_logprobs": _OUTPUT_TOKEN_LOGPROBS,
            "finish_reason": {"type": "stop"},
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(_OUTPUT_TOKEN_LOGPROBS),
        },
    }


def build_mock_sglang_app() -> FastAPI:
    """Create a minimal mock SGLang server exposing /health and /generate."""
    app = FastAPI()
    app.state.request_log: list[dict[str, Any]] = []
    app.state._log_lock = threading.Lock()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        with app.state._log_lock:
            app.state.request_log.append(body)
        payload = _generate_payload(body.get("input_ids", []) or [])
        if body.get("stream"):

            def gen_stream():
                # SGLang streams cumulative payloads; one chunk + [DONE] suffices.
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen_stream(), media_type="text/event-stream")
        return JSONResponse(content=payload)

    return app


def _reserve_local_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


class MockSGLangServer:
    """Run a mock SGLang server in a background thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port
        self.app = build_mock_sglang_app()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def request_log(self) -> list[dict[str, Any]]:
        return self.app.state.request_log

    def start(self) -> None:
        if self.port == 0:
            self.port = _reserve_local_port(self.host)
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("Mock SGLang server failed to start")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)

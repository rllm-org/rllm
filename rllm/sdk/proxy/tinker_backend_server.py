"""Lightweight FastAPI backend server wrapping TinkerEngine as an OpenAI-compatible endpoint.

This server sits between the LiteLLM proxy and TinkerEngine, translating
OpenAI chat completion requests into ``rollout_engine.get_model_response()``
calls and returning responses with embedded token IDs and logprobs in the
format expected by ``data_process.py`` extractors.

Architecture::

    SDK client -> LiteLLM proxy (metadata routing, TracingCallback)
        -> TinkerBackendServer -> TinkerEngine

The LiteLLM proxy routes to this server using the ``hosted_vllm/`` model
prefix so it treats the backend as a vLLM-compatible endpoint and passes
through ``provider_specific_fields``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

if TYPE_CHECKING:
    from rllm.engine.rollout.rollout_engine import RolloutEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class _ChatMessage(BaseModel):
    role: str
    content: str | None = None


class _ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[_ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: list[str] | str | None = None
    # LiteLLM may forward extra fields; accept silently
    extra_body: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# TinkerBackendServer
# ---------------------------------------------------------------------------


class TinkerBackendServer:
    """FastAPI server that wraps a TinkerEngine behind ``/v1/chat/completions``.

    The response embeds token IDs and logprobs so that after LiteLLM's
    ``convert_to_model_response_object()`` they end up where
    ``data_process.py`` extractors expect:

    - ``prompt_token_ids`` at response root level (non-standard root field,
      preserved via ``setattr`` by LiteLLM)
    - ``token_ids`` and ``response_logprobs`` as top-level choice fields
      (LiteLLM auto-collects non-standard choice fields into
      ``choices[].provider_specific_fields``)

    No trace storage is performed here -- that is handled by the LiteLLM
    proxy's ``TracingCallback``.

    Lifecycle follows the same pattern as ``InferenceAPIServer``: runs in a
    background daemon thread with its own asyncio event loop.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        host: str = "127.0.0.1",
        port: int = 8090,
        model_name: str = "default",
    ) -> None:
        self.rollout_engine = rollout_engine
        self.host = host
        self.port = port
        self.model_name = model_name

        self._app = self._build_app()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def base_url(self) -> str:
        return f"{self.url}/v1"

    # ------------------------------------------------------------------
    # FastAPI app
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="rLLM Tinker Backend", version="0.1.0")

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        # LiteLLM hits /v1/chat/completions on the backend
        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request, body: _ChatCompletionRequest):
            return await self._handle(body)

        # Also accept without /v1 prefix (LiteLLM may strip it)
        @app.post("/chat/completions")
        async def chat_completions_alt(request: Request, body: _ChatCompletionRequest):
            return await self._handle(body)

        return app

    async def _handle(self, body: _ChatCompletionRequest) -> dict[str, Any]:
        messages = [{"role": m.role, "content": m.content} for m in body.messages]

        kwargs: dict[str, Any] = {}
        if body.temperature is not None:
            kwargs["temperature"] = body.temperature
        if body.top_p is not None:
            kwargs["top_p"] = body.top_p
        if body.max_tokens is not None:
            kwargs["max_tokens"] = body.max_tokens
        if body.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = body.max_completion_tokens
        if body.stop is not None:
            kwargs["stop"] = body.stop

        try:
            model_output = await self.rollout_engine.get_model_response(messages, **kwargs)
        except Exception as exc:
            logger.exception("TinkerBackendServer inference error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        response_text = model_output.content or model_output.text or ""
        prompt_ids = model_output.prompt_ids or []
        completion_ids = model_output.completion_ids or []
        logprobs = model_output.logprobs or []
        finish_reason = model_output.finish_reason or "stop"

        response_message: dict[str, Any] = {"role": "assistant", "content": response_text}
        if model_output.reasoning:
            response_message["reasoning"] = model_output.reasoning

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model or self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": finish_reason,
                    # token_ids and response_logprobs must be top-level choice
                    # fields (not nested in provider_specific_fields) because
                    # LiteLLM's convert_to_model_response_object collects
                    # non-standard choice fields into provider_specific_fields
                    # automatically; a nested dict would be silently dropped.
                    "token_ids": completion_ids,
                    "response_logprobs": logprobs,
                }
            ],
            "usage": {
                "prompt_tokens": model_output.prompt_length or len(prompt_ids),
                "completion_tokens": model_output.completion_length or len(completion_ids),
                "total_tokens": (model_output.prompt_length or len(prompt_ids)) + (model_output.completion_length or len(completion_ids)),
            },
            "prompt_token_ids": prompt_ids,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the server in a background daemon thread (non-blocking)."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("TinkerBackendServer is already running")
            return

        uv_config = uvicorn.Config(
            app=self._app,
            host=self.host,
            port=self.port,
            log_level="warning",
            loop="asyncio",
        )
        self._server = uvicorn.Server(uv_config)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        if not self._started.wait(timeout=30):
            raise TimeoutError("TinkerBackendServer did not start within 30s")

        # Wait briefly for uvicorn to actually bind the socket
        self._wait_for_ready(timeout=15.0)
        logger.info("TinkerBackendServer started at %s", self.url)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._started.set()
        loop.run_until_complete(self._server.serve())

    def _wait_for_ready(self, timeout: float = 15.0) -> None:
        """Poll /health until the server is accepting connections."""
        import requests as _requests

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = _requests.get(f"{self.url}/health", timeout=0.5)
                if resp.status_code == 200:
                    return
            except _requests.RequestException:
                pass
            time.sleep(0.3)
        raise TimeoutError(f"TinkerBackendServer not ready within {timeout}s")

    def stop(self) -> None:
        """Gracefully stop the server."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("TinkerBackendServer stopped")

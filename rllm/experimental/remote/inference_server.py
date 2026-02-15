"""
Inference API Server for Remote Agent Support.

Wraps a RolloutEngine as an OpenAI-compatible HTTP API so that remote agents
(running in Docker containers or external runtimes) can query the training model
for inference during online RL episode generation.

The server exposes:
  - POST /v1/chat/completions  (OpenAI chat completions format -- text only)
  - POST /v1/model_response    (native rLLM format -- full ModelOutput with logprobs)
  - GET  /health               (health check)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from rllm.experimental.rollout.rollout_engine import RolloutEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for OpenAI-compatible request / response
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI ChatCompletion request that we support."""

    model: str = ""
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: list[str] | str | None = None
    # Pass-through for any extra sampling params the caller may set
    extra_params: dict[str, Any] = Field(default_factory=dict)
    # rLLM-specific: allow callers to tag requests for debugging / routing
    application_id: str | None = None


class ModelResponseRequest(BaseModel):
    """Request for the native rLLM model response endpoint.

    Unlike the OpenAI chat completions endpoint, this returns the full
    ``ModelOutput`` (including prompt_ids, completion_ids, logprobs, etc.)
    needed for RL training.
    """

    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: list[str] | str | None = None
    extra_params: dict[str, Any] = Field(default_factory=dict)
    application_id: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------


@dataclass
class InferenceServerConfig:
    host: str = "0.0.0.0"
    port: int = 8089
    log_level: str = "warning"


# ---------------------------------------------------------------------------
# InferenceAPIServer
# ---------------------------------------------------------------------------


class InferenceAPIServer:
    """FastAPI server that wraps a RolloutEngine as an OpenAI-compatible API.

    The server shares the same ``RolloutEngine`` instance used by the trainer,
    so any weight updates (e.g. after ``update_policy``) are immediately
    reflected in the inference responses without an explicit reload.

    Lifecycle is managed by the trainer:
        server = InferenceAPIServer(rollout_engine, config)
        server.start()   # non-blocking, runs in a background thread
        ...
        server.stop()    # graceful shutdown
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        config: InferenceServerConfig | None = None,
    ):
        self.rollout_engine = rollout_engine
        self.config = config or InferenceServerConfig()

        self._app = self._build_app()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    @property
    def url(self) -> str:
        """Base URL of the running server (e.g. ``http://0.0.0.0:8089``)."""
        return f"http://{self.config.host}:{self.config.port}"

    @property
    def inference_api_url(self) -> str:
        """URL that remote agents should use as their ``base_url``."""
        return f"{self.url}/v1"

    # ------------------------------------------------------------------
    # FastAPI app construction
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="rLLM Inference API", version="0.1.0")

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def chat_completions(request: ChatCompletionRequest):
            return await self._handle_chat_completions(request)

        @app.post("/v1/model_response")
        async def model_response(request: ModelResponseRequest):
            return await self._handle_model_response(request)

        return app

    def _extract_kwargs(self, request: ChatCompletionRequest | ModelResponseRequest) -> dict[str, Any]:
        """Extract sampling kwargs from a request object."""
        kwargs: dict[str, Any] = {}
        if request.application_id is not None:
            kwargs["application_id"] = request.application_id
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = request.max_completion_tokens
        if request.stop is not None:
            kwargs["stop"] = request.stop
        kwargs.update(request.extra_params)
        return kwargs

    async def _handle_chat_completions(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Translate an OpenAI-format request into a RolloutEngine call."""
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        kwargs = self._extract_kwargs(request)

        try:
            model_output = await self.rollout_engine.get_model_response(messages, **kwargs)
        except Exception as e:
            logger.exception(f"Inference error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

        # Map ModelOutput -> OpenAI ChatCompletion response
        response_text = model_output.content or model_output.text or ""
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason=model_output.finish_reason or "stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=model_output.prompt_length,
                completion_tokens=model_output.completion_length,
                total_tokens=model_output.prompt_length + model_output.completion_length,
            ),
        )

    async def _handle_model_response(
        self,
        request: ModelResponseRequest,
    ) -> dict:
        """Return the full ModelOutput from the RolloutEngine.

        Unlike the chat completions endpoint, this preserves all fields
        (prompt_ids, completion_ids, logprobs, etc.) needed for RL training.
        """
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        kwargs = self._extract_kwargs(request)

        try:
            model_output = await self.rollout_engine.get_model_response(messages, **kwargs)
        except Exception as e:
            logger.exception(f"Inference error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

        return {"model_output": model_output.to_dict()}

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the server in a background thread (non-blocking)."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("InferenceAPIServer is already running")
            return

        uv_config = uvicorn.Config(
            app=self._app,
            host=self.config.host,
            port=self.config.port,
            log_level=self.config.log_level,
            loop="asyncio",
        )
        self._server = uvicorn.Server(uv_config)

        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        self._started.wait(timeout=30)
        logger.info(f"InferenceAPIServer started at {self.url}")

    def _run_server(self) -> None:
        """Entry point for the background server thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Signal that the server loop is ready
        self._started.set()
        loop.run_until_complete(self._server.serve())

    def stop(self) -> None:
        """Gracefully stop the server."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("InferenceAPIServer stopped")

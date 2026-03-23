"""Lightweight proxy server that calls TinkerEngine directly.

Replaces the dual-hop architecture (LiteLLM proxy → TinkerBackendServer)
with a single HTTP hop that handles metadata-slug routing, inference, and
trace persistence in one process.

Architecture::

    Agent openai.OpenAI(base_url)
        → HTTP → TinkerProxy (port 4000)
            → TinkerEngine.get_model_response()   [in-process, no HTTP]

Per-request flow:
1. Parse metadata from ``/meta/{slug}/...`` path (if present)
2. Run inference via ``rollout_engine.get_model_response()``
3. Persist trace via ``tracer.log_llm_call()``
4. Return OpenAI-compatible response
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
from fastapi.responses import JSONResponse

from rllm.sdk.proxy.metadata_slug import extract_metadata_from_path
from rllm.sdk.proxy.tinker_backend_server import _ChatCompletionRequest

if TYPE_CHECKING:
    from rllm.engine.rollout.rollout_engine import RolloutEngine
    from rllm.sdk.tracers import SqliteTracer

logger = logging.getLogger(__name__)


class TinkerProxy:
    """Lightweight FastAPI server that calls TinkerEngine directly.

    Handles:
    - ``POST /v1/chat/completions`` — standard OpenAI path
    - ``POST /meta/{slug}/v1/chat/completions`` — metadata-slug path
    - ``POST /admin/flush-tracer`` — flush tracer queue
    - ``GET /health`` and ``GET /`` — health / readiness checks

    Lifecycle follows the same daemon-thread pattern as ``TinkerBackendServer``.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        model_name: str,
        tracer: SqliteTracer,
        host: str = "127.0.0.1",
        port: int = 4000,
        sync_tracer: bool = False,
    ) -> None:
        self.rollout_engine = rollout_engine
        self.model_name = model_name
        self.tracer = tracer
        self.host = host
        self.port = port
        self.sync_tracer = sync_tracer

        self._app = self._build_app()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ------------------------------------------------------------------
    # FastAPI app
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="rLLM TinkerProxy", version="0.1.0")

        @app.get("/")
        async def root():
            return {"status": "ok"}

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.post("/admin/flush-tracer")
        async def flush_tracer(request: Request):
            body = await request.json() if request.headers.get("content-type") == "application/json" else {}
            timeout = body.get("timeout", 30.0) if isinstance(body, dict) else 30.0
            ok = self.tracer.flush(timeout=timeout)
            return {"status": "ok" if ok else "timeout"}

        # Catch-all for /v1/chat/completions and /meta/{slug}/v1/chat/completions
        @app.api_route("/v1/chat/completions", methods=["POST"])
        async def chat_completions(request: Request):
            return await self._handle_request(request, metadata={})

        @app.api_route("/{path:path}", methods=["POST"])
        async def catch_all(request: Request, path: str):
            full_path = f"/{path}"
            # Check for metadata slug in path
            extracted = extract_metadata_from_path(full_path)
            if extracted is not None:
                cleaned_path, metadata = extracted
                # Only handle chat completions endpoint
                if cleaned_path.rstrip("/").endswith("/chat/completions"):
                    return await self._handle_request(request, metadata=metadata)

            return JSONResponse(status_code=404, content={"error": f"Not found: {full_path}"})

        return app

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    async def _handle_request(self, request: Request, metadata: dict[str, Any]) -> JSONResponse:
        """Handle a chat completion request: inference → trace → response."""
        raw_body = await request.json()
        body = _ChatCompletionRequest(**raw_body)

        t0 = time.time()

        # Run inference
        messages = [{"role": m.role, "content": m.flat_content()} for m in body.messages]

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
            logger.exception("TinkerProxy inference error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        elapsed_ms = (time.time() - t0) * 1000.0

        # Build response
        response_text = model_output.content or model_output.text or ""
        prompt_ids = model_output.prompt_ids or []
        completion_ids = model_output.completion_ids or []
        logprobs = model_output.logprobs or []
        finish_reason = model_output.finish_reason or "stop"
        prompt_len = model_output.prompt_length or len(prompt_ids)
        completion_len = model_output.completion_length or len(completion_ids)

        response_message: dict[str, Any] = {"role": "assistant", "content": response_text}
        if model_output.reasoning:
            response_message["reasoning"] = model_output.reasoning

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model_name = body.model or self.model_name

        # HTTP response keeps token_ids/response_logprobs as top-level choice
        # fields so OpenAI clients pass them through unchanged.
        response_dict: dict[str, Any] = {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": finish_reason,
                    "token_ids": completion_ids,
                    "response_logprobs": logprobs,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_len,
                "total_tokens": prompt_len + completion_len,
            },
            "prompt_token_ids": prompt_ids,
        }

        # Trace output nests token_ids/response_logprobs under
        # provider_specific_fields (matching LiteLLM's ModelResponse format
        # that data_process.py extractors expect).
        trace_output: dict[str, Any] = {
            "id": response_id,
            "object": "chat.completion",
            "created": response_dict["created"],
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": finish_reason,
                    "provider_specific_fields": {
                        "token_ids": completion_ids,
                        "response_logprobs": logprobs,
                    },
                }
            ],
            "usage": response_dict["usage"],
            "prompt_token_ids": prompt_ids,
        }

        # Persist trace
        tokens = {
            "prompt": prompt_len,
            "completion": completion_len,
            "total": prompt_len + completion_len,
        }

        log_kwargs = dict(
            name=f"proxy/{model_name}",
            model=model_name,
            input={"messages": messages},
            output=trace_output,
            metadata=metadata,
            session_name=metadata.get("session_name"),
            session_uids=metadata.get("session_uids"),
            latency_ms=elapsed_ms,
            tokens=tokens,
            trace_id=response_id,
        )

        if self.sync_tracer:
            await self.tracer.log_llm_call_sync(**log_kwargs)
        else:
            self.tracer.log_llm_call(**log_kwargs)

        return JSONResponse(content=response_dict)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the server in a background daemon thread (non-blocking)."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("TinkerProxy is already running")
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
            raise TimeoutError("TinkerProxy did not start within 30s")

        self._wait_for_ready(timeout=15.0)
        logger.info("TinkerProxy started at %s", self.url)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._started.set()
        loop.run_until_complete(self._server.serve())

    def _wait_for_ready(self, timeout: float = 15.0) -> None:
        """Poll / until the server is accepting connections."""
        import requests as _requests

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = _requests.get(f"{self.url}/", timeout=0.5)
                if resp.status_code == 200:
                    return
            except _requests.RequestException:
                pass
            time.sleep(0.3)
        raise TimeoutError(f"TinkerProxy not ready within {timeout}s")

    def stop(self) -> None:
        """Gracefully stop the server."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("TinkerProxy stopped")

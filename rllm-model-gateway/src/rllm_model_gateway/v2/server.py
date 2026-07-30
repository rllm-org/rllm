import argparse
import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from rllm_model_gateway.v2.auth import GatewayAuth
from rllm_model_gateway.v2.config import BackendConfig, GatewayConfig, TokenizationConfig
from rllm_model_gateway.v2.contracts import APIProtocol, CanonicalOutput, SessionTraces
from rllm_model_gateway.v2.errors import GatewayError
from rllm_model_gateway.v2.pool import WorkerPool
from rllm_model_gateway.v2.protocols import error_payload, normalize_request, response_payload, stream_events


class SessionCreateRequest(BaseModel):
    session_id: str | None = None
    sampling_params: dict[str, Any] = Field(default_factory=dict)


def create_app(config: GatewayConfig, pool: WorkerPool | None = None) -> FastAPI:
    worker_pool = pool or WorkerPool(config)
    auth = GatewayAuth(config.admin_key)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        worker_pool.start()
        try:
            yield
        finally:
            worker_pool.stop()

    app = FastAPI(title="rllm-model-gateway", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(GatewayError)
    async def handle_gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=error_payload(str(exc), exc.error_type))

    def require_admin(authorization: str | None) -> None:
        auth.require_admin(authorization)

    def require_agent(authorization: str | None, session_id: str) -> None:
        auth.require_session(authorization, session_id)

    @app.get("/health")
    async def health() -> dict[str, str]:
        worker_pool.health()
        return {"status": "ok"}

    @app.post("/admin/sessions")
    async def create_session(body: SessionCreateRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        require_admin(authorization)
        session_id = body.session_id or f"session_{uuid.uuid4().hex}"
        await worker_pool.create_session(
            session_id,
            {
                "session_id": session_id,
                "sampling_params": body.sampling_params,
            },
        )
        agent_key = auth.issue_session_key(session_id)
        base_url = config.agent_base_url.rstrip("/")
        session_path = quote(session_id, safe="/")
        return {"session_id": session_id, "url": f"{base_url}/sessions/{session_path}/v1", "agent_key": agent_key}

    @app.get("/admin/sessions/{session_id:path}")
    async def get_session(session_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        require_admin(authorization)
        payload = await worker_pool.call(session_id, "get_session_traces", {"session_id": session_id})
        return SessionTraces.model_validate(payload).model_dump(mode="json")

    @app.delete("/admin/sessions/{session_id:path}")
    async def delete_session(session_id: str, authorization: str | None = Header(default=None)) -> Response:
        require_admin(authorization)
        auth.revoke_session(session_id)
        await worker_pool.delete_session(session_id, {"session_id": session_id})
        return Response(status_code=204)

    async def inference(session_id: str, protocol: APIProtocol, request: Request, authorization: str | None) -> Response:
        require_agent(authorization, session_id)
        try:
            body = await request.json()
        except Exception as exc:
            raise GatewayError("request body must be valid JSON") from exc
        if not isinstance(body, dict):
            raise GatewayError("request body must be a JSON object")
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise GatewayError("stream must be a boolean")
        stream_options = body.get("stream_options")
        if stream_options is not None and not isinstance(stream_options, dict):
            raise GatewayError("stream_options must be an object")
        if stream_options is not None and not stream:
            raise GatewayError("stream_options may only be used when stream is true")
        include_usage = stream_options.get("include_usage", False) if stream_options else False
        if not isinstance(include_usage, bool):
            raise GatewayError("stream_options.include_usage must be a boolean")
        canonical_request = normalize_request(protocol, session_id, body)
        response_model = str(body.get("model") or "")
        created_at = int(time.time())

        async def run() -> CanonicalOutput:
            payload = await worker_pool.call(
                session_id,
                "generate",
                {"request": canonical_request.model_dump(mode="json")},
            )
            return CanonicalOutput.model_validate(payload)

        if not stream:
            task = asyncio.create_task(run())
            try:
                result = await asyncio.wait_for(asyncio.shield(task), timeout=config.heartbeat_seconds)
                return JSONResponse(content=response_payload(protocol, result, response_model, created_at))
            except TimeoutError:
                pass

            async def json_with_heartbeat() -> AsyncIterator[str]:
                while True:
                    try:
                        result = await asyncio.wait_for(asyncio.shield(task), timeout=config.heartbeat_seconds)
                        break
                    except TimeoutError:
                        yield " "
                    except GatewayError as exc:
                        yield json.dumps(error_payload(str(exc), exc.error_type), separators=(",", ":"))
                        return
                    except Exception as exc:
                        yield json.dumps(error_payload(str(exc)), separators=(",", ":"))
                        return
                yield json.dumps(response_payload(protocol, result, response_model, created_at), separators=(",", ":"))

            return StreamingResponse(
                json_with_heartbeat(),
                media_type="application/json",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        async def fake_stream() -> AsyncIterator[str]:
            task = asyncio.create_task(run())
            while True:
                try:
                    result = await asyncio.wait_for(asyncio.shield(task), timeout=config.heartbeat_seconds)
                    break
                except TimeoutError:
                    yield ": keepalive\n\n"
                except GatewayError as exc:
                    yield f"data: {json.dumps(error_payload(str(exc), exc.error_type), separators=(',', ':'))}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                except Exception as exc:
                    yield f"data: {json.dumps(error_payload(str(exc)), separators=(',', ':'))}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            for event in stream_events(protocol, result, response_model, created_at, include_usage):
                yield event

        return StreamingResponse(fake_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/sessions/{session_id:path}/v1/completions")
    async def completions(session_id: str, request: Request, authorization: str | None = Header(default=None)) -> Response:
        return await inference(session_id, APIProtocol.COMPLETIONS, request, authorization)

    @app.post("/sessions/{session_id:path}/v1/chat/completions")
    async def chat_completions(session_id: str, request: Request, authorization: str | None = Header(default=None)) -> Response:
        return await inference(session_id, APIProtocol.CHAT_COMPLETIONS, request, authorization)

    app.state.config = config
    app.state.worker_pool = worker_pool
    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the rLLM model gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--admin-key", required=True)
    parser.add_argument("--agent-base-url", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--backend-kwargs-json", default="{}")
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--renderer", default="auto")
    parser.add_argument("--renderer-kwargs-json", default="{}")
    parser.add_argument("--cumulative", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", default="info")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        backend_kwargs = json.loads(args.backend_kwargs_json)
    except json.JSONDecodeError as exc:
        parser.error(f"--backend-kwargs-json must be valid JSON: {exc}")
    if not isinstance(backend_kwargs, dict):
        parser.error("--backend-kwargs-json must decode to an object")
    try:
        renderer_kwargs = json.loads(args.renderer_kwargs_json)
    except json.JSONDecodeError as exc:
        parser.error(f"--renderer-kwargs-json must be valid JSON: {exc}")
    if not isinstance(renderer_kwargs, dict):
        parser.error("--renderer-kwargs-json must decode to an object")
    config = GatewayConfig(
        host=args.host,
        port=args.port,
        num_workers=args.workers,
        admin_key=args.admin_key,
        agent_base_url=args.agent_base_url,
        cumulative=args.cumulative,
        tokenization=TokenizationConfig(
            model=args.tokenizer_model,
            trust_remote_code=args.trust_remote_code,
            renderer=args.renderer,
            renderer_kwargs=renderer_kwargs,
        ),
        backend=BackendConfig(
            name=args.backend,
            kwargs=backend_kwargs,
        ),
    )
    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level=args.log_level)

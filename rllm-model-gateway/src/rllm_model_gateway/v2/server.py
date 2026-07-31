import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from rllm_model_gateway.v2.auth import GatewayAuth
from rllm_model_gateway.v2.config import GatewayConfig
from rllm_model_gateway.v2.inference import InferenceClientClass
from rllm_model_gateway.v2.pool import WorkerPool
from rllm_model_gateway.v2.protocols import error_payload, normalize_request, response_payload, stream_events
from rllm_model_gateway.v2.types import APIProtocol, GatewayError, GatewayResponse, SessionTraces


class SessionCreateRequest(BaseModel):
    session_id: str | None = None
    sampling_params: dict[str, Any] = Field(default_factory=dict)


def create_app(
    config: GatewayConfig,
    inference_client_cls: InferenceClientClass,
    inference_client_kwargs: dict[str, Any],
    gateway_connection: Any,
    shutdown: Callable[[], None],
) -> FastAPI:
    worker_pool = WorkerPool(config, inference_client_cls, inference_client_kwargs)
    auth = GatewayAuth(config.admin_key)

    async def process_control_requests() -> None:
        while True:
            try:
                if not await asyncio.to_thread(gateway_connection.poll, 0.5):
                    continue
                update = await asyncio.to_thread(gateway_connection.recv)
            except EOFError:
                return
            if update is None:
                shutdown()
                return
            try:
                await worker_pool.update_inference_client(update)
                response = {"ok": True}
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            try:
                await asyncio.to_thread(gateway_connection.send, response)
            except (BrokenPipeError, EOFError):
                return

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        worker_pool.start()
        control_task = asyncio.create_task(process_control_requests())
        try:
            yield
        finally:
            control_task.cancel()
            await asyncio.gather(control_task, return_exceptions=True)
            gateway_connection.close()
            worker_pool.stop()

    app = FastAPI(title="rllm-model-gateway", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(GatewayError)
    async def handle_gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(str(exc), exc.error_type, exc.status_code),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_payload(str(exc), "invalid_request_error", 400),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        message = str(exc) or exc.__class__.__name__
        return JSONResponse(status_code=500, content=error_payload(message, "server_error", 500))

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
        return {"session_id": session_id, "agent_key": agent_key}

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

        async def run() -> GatewayResponse:
            payload = await worker_pool.call(
                session_id,
                "generate",
                {"request": canonical_request.model_dump(mode="json")},
            )
            return GatewayResponse.model_validate(payload)

        if not stream:
            task = asyncio.create_task(run())
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=config.heartbeat_initial_delay_seconds,
                )
                return JSONResponse(content=response_payload(protocol, result, response_model, created_at))
            except TimeoutError:
                pass

            async def json_with_heartbeat() -> AsyncIterator[str]:
                yield " "
                while True:
                    try:
                        result = await asyncio.wait_for(
                            asyncio.shield(task),
                            timeout=config.heartbeat_interval_seconds,
                        )
                        break
                    except TimeoutError:
                        yield " "
                    except GatewayError as exc:
                        yield json.dumps(
                            error_payload(str(exc), exc.error_type, exc.status_code),
                            separators=(",", ":"),
                        )
                        return
                    except Exception as exc:
                        yield json.dumps(error_payload(str(exc), code=500), separators=(",", ":"))
                        return
                yield json.dumps(response_payload(protocol, result, response_model, created_at), separators=(",", ":"))

            return StreamingResponse(
                json_with_heartbeat(),
                media_type="application/json",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        async def fake_stream() -> AsyncIterator[str]:
            task = asyncio.create_task(run())
            wait_seconds = config.heartbeat_initial_delay_seconds
            while True:
                try:
                    result = await asyncio.wait_for(asyncio.shield(task), timeout=wait_seconds)
                    break
                except TimeoutError:
                    yield ": keepalive\n\n"
                    wait_seconds = config.heartbeat_interval_seconds
                except GatewayError as exc:
                    yield f"data: {json.dumps(error_payload(str(exc), exc.error_type, exc.status_code), separators=(',', ':'))}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                except Exception as exc:
                    yield f"data: {json.dumps(error_payload(str(exc), code=500), separators=(',', ':'))}\n\n"
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

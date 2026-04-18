"""FastAPI app + CLI for rllm-model-gateway.

Picks one of two modes at startup:

  passthrough — no adapter; forwards every request to ``config.upstream_url``.
                The agent's auth header is stripped and (optionally) replaced
                with ``config.upstream_api_key``. Upstream's response is
                returned to the client as-is.

  adapter     — adapter callable passed to ``create_app(adapter=...)``.
                Wire request → NormalizedRequest → adapter → NormalizedResponse
                → wire response.

In both modes per-session sampling params and ``config.model`` can override
the request body. Traces persist to TraceStore for every call.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import json
import logging
import os
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from rllm_model_gateway.auth import AuthMiddleware
from rllm_model_gateway.config import GatewayConfig
from rllm_model_gateway.endpoints import SHAPERS
from rllm_model_gateway.middleware import SessionRoutingMiddleware
from rllm_model_gateway.normalized import AdapterError, NormalizedRequest, NormalizedResponse
from rllm_model_gateway.store.base import TraceStore
from rllm_model_gateway.trace import build_trace, serialize_extras

logger = logging.getLogger(__name__)


AdapterFn = Callable[[NormalizedRequest], Awaitable[NormalizedResponse]]


# ---------------------------------------------------------------------------
# Store factory
# ---------------------------------------------------------------------------


def _create_store(config: GatewayConfig) -> TraceStore:
    """Default store factory. Tests override by passing ``store=...`` to create_app."""
    from rllm_model_gateway.store.sqlite_store import SqliteTraceStore

    return SqliteTraceStore(db_path=config.db_path)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    config: GatewayConfig | None = None,
    store: TraceStore | None = None,
    adapter: AdapterFn | None = None,
) -> FastAPI:
    """Build the gateway FastAPI app.

    Args:
        config: Gateway configuration.
        store: Trace store. Defaults to a SQLite store rooted at config.db_path.
        adapter: When set, the gateway runs in adapter mode and calls this
            function for every request. When unset, the gateway runs in
            passthrough mode and forwards to ``config.upstream_url``.
    """
    if config is None:
        config = GatewayConfig()
    if store is None:
        store = _create_store(config)

    if adapter is None and not config.upstream_url:
        raise ValueError("Gateway requires either an adapter (via create_app(adapter=...)) or config.upstream_url (passthrough mode).")
    if adapter is not None and not inspect.iscoroutinefunction(adapter):
        raise TypeError(f"adapter must be `async def`; got {type(adapter).__name__}. Wrap with `async def adapter(req): return ...`.")
    if adapter is not None and config.upstream_url:
        logger.warning("Both adapter and upstream_url configured; adapter takes precedence (upstream_url ignored).")

    # Auto-generate API keys if not provided. Both keys are logged at startup
    # (visible in the engine's training logs) — auto-generated keys reset on
    # restart, so pin them via config / env if you need stability.
    admin_was_provided = config.admin_api_key is not None
    agent_was_provided = config.agent_api_key is not None
    if not admin_was_provided:
        config.admin_api_key = secrets.token_urlsafe(32)
    if not agent_was_provided:
        config.agent_api_key = secrets.token_urlsafe(32)
    logger.info(
        "Gateway API keys: admin=%s%s  agent=%s%s",
        config.admin_api_key,
        "" if admin_was_provided else " (auto-generated)",
        config.agent_api_key,
        "" if agent_was_provided else " (auto-generated)",
    )

    pending_traces: set[asyncio.Task] = set()
    http_client_holder: dict[str, httpx.AsyncClient | None] = {"client": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if adapter is None:
            http_client_holder["client"] = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=None),
                limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
                follow_redirects=True,
            )
        yield
        if pending_traces:
            await asyncio.gather(*pending_traces, return_exceptions=True)
            pending_traces.clear()
        if http_client_holder["client"] is not None:
            await http_client_holder["client"].aclose()
            http_client_holder["client"] = None
        await store.close()

    app = FastAPI(title="rllm-model-gateway", version="0.2.0", lifespan=lifespan)
    # Order: session routing rewrites the path FIRST, but we need the original
    # path for auth classification (proxy vs management). Starlette runs
    # middleware in reverse-add order, so add session-routing first (runs
    # last → after auth) and auth second (runs first → sees original path).
    app.add_middleware(SessionRoutingMiddleware)
    app.add_middleware(
        AuthMiddleware,
        admin_api_key=config.admin_api_key,
        agent_api_key=config.agent_api_key,
    )

    # -- Health --------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # -- Sessions ------------------------------------------------------

    @app.post("/sessions")
    async def create_session(request: Request):
        body = await _safe_json(request)
        sid = body.get("session_id") or str(uuid.uuid4())
        await store.create_session(
            session_id=sid,
            metadata=body.get("metadata"),
            sampling_params=body.get("sampling_params"),
        )
        return {"session_id": sid, "url": f"/sessions/{sid}/v1"}

    @app.get("/sessions")
    async def list_sessions(
        since: float | None = Query(None),
        limit: int | None = Query(None),
    ):
        return await store.list_sessions(since=since, limit=limit)

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        info = await store.get_session(session_id)
        if info is None:
            return _error(404, f"Session {session_id!r} not found")
        return info

    @app.get("/sessions/{session_id}/traces")
    async def get_session_traces(
        session_id: str,
        since: float | None = Query(None),
        limit: int | None = Query(None),
    ):
        traces = await store.get_traces(session_id, since=since, limit=limit)
        return [t.model_dump(mode="json") for t in traces]

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        count = await store.delete_session(session_id)
        return {"deleted": count}

    # -- Traces --------------------------------------------------------

    @app.get("/traces/{trace_id}")
    async def get_trace(trace_id: str):
        trace = await store.get_trace(trace_id)
        if trace is None:
            return _error(404, f"Trace {trace_id!r} not found")
        return trace.model_dump(mode="json")

    @app.get("/traces/{trace_id}/extras")
    async def get_trace_extras(trace_id: str):
        extras = await store.get_trace_extras(trace_id)
        if extras is None:
            return _error(404, f"No extras for trace {trace_id!r}")
        fmt, data = extras
        return Response(content=data, media_type=f"application/x-{fmt}", headers={"X-Extras-Format": fmt})

    @app.post("/admin/flush")
    async def flush():
        # Drain any in-flight async trace writes, then flush the store.
        if pending_traces:
            await asyncio.gather(*pending_traces, return_exceptions=True)
        await store.flush()
        return {"status": "flushed"}

    # -- Endpoint routes -----------------------------------------------

    for shaper in SHAPERS.values():
        _attach_endpoint_route(
            app=app,
            store=store,
            shaper_name=shaper.NAME,
            path=shaper.PATH,
            adapter=adapter,
            config=config,
            http_client_holder=http_client_holder,
            pending_traces=pending_traces,
        )

    app.state.config = config
    app.state.store = store

    return app


# ---------------------------------------------------------------------------
# Endpoint route handler
# ---------------------------------------------------------------------------


def _attach_endpoint_route(
    *,
    app: FastAPI,
    store: TraceStore,
    shaper_name: str,
    path: str,
    adapter: AdapterFn | None,
    config: GatewayConfig,
    http_client_holder: dict[str, httpx.AsyncClient | None],
    pending_traces: set,
):
    async def handler(request: Request):
        return await _handle_request(
            request=request,
            store=store,
            shaper_name=shaper_name,
            adapter=adapter,
            config=config,
            http_client_holder=http_client_holder,
            pending_traces=pending_traces,
        )

    handler.__name__ = f"proxy_{shaper_name}"
    app.post(path)(handler)


async def _handle_request(
    *,
    request: Request,
    store: TraceStore,
    shaper_name: str,
    adapter: AdapterFn | None,
    config: GatewayConfig,
    http_client_holder: dict[str, httpx.AsyncClient | None],
    pending_traces: set,
):
    state = request.scope.get("state", {})
    session_id = state.get("session_id")
    if not session_id:
        return _error(400, "Request must be made under /sessions/{sid}/v1/...")

    # Sessions must be created explicitly via POST /sessions before requests
    # can be made under that ID. Catches typo'd session IDs immediately.
    session_info = await store.get_session(session_id)
    if session_info is None:
        return _error(
            404,
            f"Session {session_id!r} not found. Create it with POST /sessions first.",
        )

    raw_body = await request.body()
    try:
        body_dict = json.loads(raw_body) if raw_body else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        body_dict = {}

    session_sampling = session_info.get("sampling_params") or {}
    body_dict = _apply_overrides(
        body_dict,
        session_sampling=session_sampling,
        priority=config.sampling_params_priority,
        model_pin=config.model,
    )
    raw_body = json.dumps(body_dict).encode("utf-8") if body_dict else raw_body

    stream = bool(body_dict.get("stream"))
    model = body_dict.get("model", "")
    shaper = SHAPERS[shaper_name]
    t0 = time.perf_counter()

    if adapter is not None:
        return await _handle_adapter_mode(
            adapter=adapter,
            shaper=shaper,
            shaper_name=shaper_name,
            body_dict=body_dict,
            session_id=session_id,
            model=model,
            stream=stream,
            store=store,
            pending_traces=pending_traces,
            t0=t0,
        )
    return await _handle_passthrough_mode(
        request=request,
        raw_body=raw_body,
        body_dict=body_dict,
        shaper=shaper,
        shaper_name=shaper_name,
        session_id=session_id,
        model=model,
        stream=stream,
        config=config,
        http_client_holder=http_client_holder,
        store=store,
        pending_traces=pending_traces,
        t0=t0,
    )


async def _handle_adapter_mode(
    *,
    adapter: AdapterFn,
    shaper,
    shaper_name: str,
    body_dict: dict[str, Any],
    session_id: str,
    model: str,
    stream: bool,
    store: TraceStore,
    pending_traces: set,
    t0: float,
):
    # Adapter can return only one completion. Multi-completion (n>1) requests
    # would need adapter-side fanout; we don't do that. Reject explicitly so
    # the agent doesn't silently get a single response when it asked for many.
    n = body_dict.get("n")
    if isinstance(n, int) and n > 1:
        return _error(400, f"n>1 is not supported in adapter mode (got n={n}).")

    try:
        normalized_req = shaper.to_normalized_request(body_dict)
        normalized_req.session_id = session_id
    except ValueError as exc:
        return _error(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Request shaper failed for %s", shaper_name)
        return _error(400, f"Invalid request body: {exc}")

    try:
        normalized_resp = await adapter(normalized_req)
    except AdapterError as exc:
        logger.warning("Adapter signaled %d: %s", exc.status_code, exc.message)
        return _error(exc.status_code, exc.message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Adapter raised")
        return _error(502, f"Adapter error: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000
    _spawn_persist(
        store,
        session_id,
        shaper_name,
        model,
        normalized_req,
        normalized_resp,
        latency_ms,
        pending_traces,
    )

    if stream:
        agen = shaper.from_normalized_response_stream(normalized_resp, model)
        return StreamingResponse(agen, media_type="text/event-stream")
    wire = shaper.from_normalized_response_nonstream(normalized_resp, model)
    return JSONResponse(content=wire, status_code=200)


async def _handle_passthrough_mode(
    *,
    request: Request,
    raw_body: bytes,
    body_dict: dict[str, Any],
    shaper,
    shaper_name: str,
    session_id: str,
    model: str,
    stream: bool,
    config: GatewayConfig,
    http_client_holder: dict[str, httpx.AsyncClient | None],
    store: TraceStore,
    pending_traces: set,
    t0: float,
):
    http = http_client_holder["client"]
    assert http is not None, "passthrough mode requires the lifespan to have started httpx"

    upstream_url = f"{config.upstream_url.rstrip('/')}{shaper.UPSTREAM_PATH}"
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"

    forward_headers = _build_passthrough_headers(request.headers, config.upstream_api_key)

    if not stream:
        try:
            resp = await _send_with_retry(
                http,
                "POST",
                upstream_url,
                raw_body,
                forward_headers,
                max_retries=config.max_retries,
            )
        except httpx.ConnectError as exc:
            logger.exception("Upstream unreachable after retries")
            return _error(502, f"Upstream unreachable: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Upstream forward failed")
            return _error(502, f"Upstream error: {exc}")
        latency_ms = (time.perf_counter() - t0) * 1000

        try:
            upstream_dict = json.loads(resp.content) if resp.content else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            upstream_dict = {}

        normalized_req_for_trace = _normalize_request_for_trace(shaper, body_dict)
        normalized_resp = NormalizedResponse()
        if isinstance(upstream_dict, dict) and upstream_dict:
            with contextlib.suppress(Exception):
                normalized_resp = shaper.parse_upstream_response(upstream_dict)

        _spawn_persist(
            store,
            session_id,
            shaper_name,
            model or upstream_dict.get("model", ""),
            normalized_req_for_trace,
            normalized_resp,
            latency_ms,
            pending_traces,
        )
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

    # Streaming passthrough: tee chunks for trace capture.
    accumulated: list[dict] = []

    async def relay() -> AsyncIterator[bytes]:
        async with http.stream("POST", upstream_url, content=raw_body, headers=forward_headers) as upstream:
            try:
                async for line in upstream.aiter_lines():
                    if not line:
                        yield b"\n"
                        continue
                    yield (line + "\n").encode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str and data_str != "[DONE]":
                            with contextlib.suppress(json.JSONDecodeError):
                                accumulated.append(json.loads(data_str))
            finally:
                latency_ms = (time.perf_counter() - t0) * 1000
                normalized_req_for_trace = _normalize_request_for_trace(shaper, body_dict)
                normalized_resp = NormalizedResponse()
                if accumulated:
                    with contextlib.suppress(Exception):
                        normalized_resp = shaper.parse_upstream_stream(accumulated)
                _spawn_persist(
                    store,
                    session_id,
                    shaper_name,
                    model,
                    normalized_req_for_trace,
                    normalized_resp,
                    latency_ms,
                    pending_traces,
                )

    return StreamingResponse(relay(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "content-length",
        "content-encoding",
        "host",
    }
)

# Auth headers presented by the agent — the gateway's own keys, meaningless
# to OpenAI/Anthropic/etc. Always stripped before forwarding upstream.
_AUTH_HEADERS_TO_STRIP = frozenset({"authorization", "x-api-key"})


def _build_passthrough_headers(
    incoming,
    upstream_api_key: str | None,
) -> dict[str, str]:
    out = {k: v for k, v in incoming.items() if k.lower() not in _HOP_BY_HOP and k.lower() not in _AUTH_HEADERS_TO_STRIP}
    if upstream_api_key:
        # Send both header conventions — works regardless of which upstream
        # we're forwarding to.
        out["Authorization"] = f"Bearer {upstream_api_key}"
        out["x-api-key"] = upstream_api_key
    return out


def _apply_overrides(
    body: dict[str, Any],
    *,
    session_sampling: dict[str, Any],
    priority: str,
    model_pin: str | None,
) -> dict[str, Any]:
    """Mutate request body with per-session sampling params + model pin."""
    if not isinstance(body, dict):
        return body
    if model_pin:
        body["model"] = model_pin
    if session_sampling:
        if priority == "session":
            body.update(session_sampling)
        else:  # "client" — request wins on conflict
            for k, v in session_sampling.items():
                body.setdefault(k, v)
    return body


def _normalize_request_for_trace(shaper, body: dict[str, Any]) -> NormalizedRequest:
    """Best-effort normalize for trace capture; never raises."""
    try:
        return shaper.to_normalized_request(body)
    except Exception:  # noqa: BLE001
        logger.exception("Request shaper failed during trace capture")
        return NormalizedRequest()


async def _send_with_retry(
    http: httpx.AsyncClient,
    method: str,
    url: str,
    content: bytes,
    headers: dict[str, str],
    max_retries: int,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            return await http.request(method, url, content=content, headers=headers)
        except httpx.ConnectError as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning("Connection error (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc)
    assert last_exc is not None
    raise last_exc


def _spawn_persist(
    store: TraceStore,
    session_id: str,
    endpoint: str,
    model: str,
    request: NormalizedRequest,
    response: NormalizedResponse,
    gateway_latency_ms: float,
    pending: set,
) -> None:
    task = asyncio.create_task(_do_persist(store, session_id, endpoint, model, request, response, gateway_latency_ms))
    pending.add(task)
    task.add_done_callback(pending.discard)


async def _do_persist(
    store: TraceStore,
    session_id: str,
    endpoint: str,
    model: str,
    request: NormalizedRequest,
    response: NormalizedResponse,
    gateway_latency_ms: float,
) -> None:
    try:
        trace = build_trace(
            session_id=session_id,
            endpoint=endpoint,
            model=model,
            request=request,
            response=response,
            gateway_latency_ms=gateway_latency_ms,
        )
        extras = serialize_extras(response.extras if response else {})
        await store.store_trace(trace, extras=extras)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to persist trace (session=%s endpoint=%s)", session_id, endpoint)


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


async def _safe_json(request: Request) -> dict[str, Any]:
    try:
        return await request.json()
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Config loading + CLI
# ---------------------------------------------------------------------------


def _load_config(args: argparse.Namespace) -> GatewayConfig:
    data: dict[str, Any] = {}

    config_path = getattr(args, "config", None)
    if config_path:
        try:
            with open(config_path) as f:
                yaml_data = yaml.safe_load(f) or {}
        except FileNotFoundError as exc:
            raise SystemExit(f"Config file not found: {config_path}") from exc
        except (OSError, yaml.YAMLError) as exc:
            raise SystemExit(f"Failed to load config {config_path!r}: {exc}") from exc
        if not isinstance(yaml_data, dict):
            raise SystemExit(f"Config {config_path!r} must contain a YAML mapping at the top level")
        data.update(yaml_data)

    env_map = {
        "RLLM_GATEWAY_HOST": "host",
        "RLLM_GATEWAY_PORT": "port",
        "RLLM_GATEWAY_DB_PATH": "db_path",
        "RLLM_GATEWAY_LOG_LEVEL": "log_level",
        "RLLM_GATEWAY_UPSTREAM_URL": "upstream_url",
        "RLLM_GATEWAY_UPSTREAM_KEY": "upstream_api_key",
        "RLLM_GATEWAY_MODEL": "model",
        "RLLM_GATEWAY_ADMIN_KEY": "admin_api_key",
        "RLLM_GATEWAY_AGENT_KEY": "agent_api_key",
    }
    for env_key, cfg_key in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            data[cfg_key] = int(val) if cfg_key == "port" else val

    if getattr(args, "host", None) is not None:
        data["host"] = args.host
    if getattr(args, "port", None) is not None:
        data["port"] = args.port
    if getattr(args, "db_path", None) is not None:
        data["db_path"] = args.db_path
    if getattr(args, "log_level", None) is not None:
        data["log_level"] = args.log_level
    if getattr(args, "upstream_url", None) is not None:
        data["upstream_url"] = args.upstream_url
    if getattr(args, "upstream_key", None) is not None:
        data["upstream_api_key"] = args.upstream_key
    if getattr(args, "model", None) is not None:
        data["model"] = args.model
    if getattr(args, "sampling_params_priority", None) is not None:
        data["sampling_params_priority"] = args.sampling_params_priority
    if getattr(args, "admin_key", None) is not None:
        data["admin_api_key"] = args.admin_key
    if getattr(args, "agent_key", None) is not None:
        data["agent_api_key"] = args.agent_key

    return GatewayConfig(**data)


def main() -> None:
    parser = argparse.ArgumentParser(description="rllm-model-gateway")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--upstream-url", type=str, default=None, dest="upstream_url", help="URL to forward requests to in passthrough mode.")
    parser.add_argument("--upstream-key", type=str, default=None, dest="upstream_key", help="API key sent to the upstream on outbound passthrough requests.")
    parser.add_argument("--model", type=str, default=None, help="If set, rewrites the request body's 'model' field on every call.")
    parser.add_argument("--sampling-params-priority", type=str, default=None, choices=["client", "session"], dest="sampling_params_priority")
    parser.add_argument("--db-path", type=str, default=None, dest="db_path")
    parser.add_argument("--log-level", type=str, default=None, dest="log_level")
    parser.add_argument("--admin-key", type=str, default=None, dest="admin_key", help="Admin API key (auto-generated if not provided).")
    parser.add_argument("--agent-key", type=str, default=None, dest="agent_key", help="Agent API key (auto-generated if not provided).")
    args = parser.parse_args()

    config = _load_config(args)
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    app = create_app(config)

    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level.lower())


if __name__ == "__main__":
    main()

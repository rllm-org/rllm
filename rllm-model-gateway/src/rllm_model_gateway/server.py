"""FastAPI application factory and CLI entrypoint for rllm-model-gateway."""

import argparse
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import yaml
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from rllm_model_gateway.middleware import SessionRoutingMiddleware
from rllm_model_gateway.models import (
    GatewayConfig,
    WorkerInfo,
)
from rllm_model_gateway.proxy import ReverseProxy
from rllm_model_gateway.session_manager import SessionManager
from rllm_model_gateway.session_router import SessionRouter
from rllm_model_gateway.store.base import TraceStore

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Store factory
# ------------------------------------------------------------------


def create_store(config: GatewayConfig) -> TraceStore:
    worker = config.store_worker
    if worker == "sqlite":
        from rllm_model_gateway.store.sqlite_store import SqliteTraceStore

        return SqliteTraceStore(db_path=config.db_path)
    elif worker == "memory":
        from rllm_model_gateway.store.memory_store import MemoryTraceStore

        return MemoryTraceStore()
    else:
        raise ValueError(f"Unknown store worker: {worker}")


# ------------------------------------------------------------------
# Routing policy loader
# ------------------------------------------------------------------


def _load_policy(dotted_path: str):
    """Import a class by dotted path, e.g. ``my_pkg.policies.CacheAwarePolicy``."""
    module_path, _, cls_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid policy path: {dotted_path}")
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)()


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------


def create_app(
    config: GatewayConfig | None = None,
    store: TraceStore | None = None,
) -> FastAPI:
    """Create and return a fully configured FastAPI application."""
    if config is None:
        config = GatewayConfig()

    if store is None:
        store = create_store(config)

    # Routing policy
    policy = None
    if config.routing_policy:
        policy = _load_policy(config.routing_policy)

    router = SessionRouter(
        policy=policy,
        health_check_interval=config.health_check_interval,
    )

    # Register initial workers
    for i, wc in enumerate(config.workers):
        router.add_worker(
            WorkerInfo(
                worker_id=wc.worker_id or str(i),
                url=wc.url,
                model_name=wc.model_name,
                weight=wc.weight,
            )
        )

    proxy = ReverseProxy(
        router=router,
        store=store,
        strip_vllm=config.strip_vllm_fields,
        sync_traces=config.sync_traces,
    )
    sessions = SessionManager(store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await proxy.start()
        if router.workers:
            await router.start_health_checks()
        yield
        await router.stop_health_checks()
        await proxy.stop()
        await store.close()

    app = FastAPI(title="rllm-model-gateway", version="0.1.0", lifespan=lifespan)

    # -- Middleware ---------------------------------------------------------

    # TODO: Add API key auth middleware here for securing gateway access
    # from cloud containers. Validate an API key from the Authorization
    # header before allowing access to admin and proxy endpoints.
    # Add corresponding `api_key` field to GatewayConfig and client classes.

    app.add_middleware(
        SessionRoutingMiddleware,
        add_logprobs=config.add_logprobs,
        add_return_token_ids=config.add_return_token_ids,
    )

    # -- Health endpoints --------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/health/workers")
    async def health_workers():
        workers = router.get_workers()
        return {
            "workers": [w.model_dump() for w in workers],
            "healthy": sum(1 for w in workers if w.healthy),
            "total": len(workers),
        }

    # -- Session endpoints -------------------------------------------------

    @app.post("/sessions")
    async def create_session(request: Request):
        body = await _safe_json(request)
        sid = sessions.create_session(
            session_id=body.get("session_id"),
            metadata=body.get("metadata"),
        )
        return {"session_id": sid, "url": f"/sessions/{sid}/v1"}

    @app.get("/sessions")
    async def list_sessions(
        since: float | None = Query(None),
        limit: int | None = Query(None),
    ):
        result = await sessions.list_sessions(since=since, limit=limit)
        return [s.model_dump() for s in result]

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        info = await sessions.get_session_info(session_id)
        if info is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Session {session_id} not found"},
            )
        return info.model_dump()

    @app.get("/sessions/{session_id}/traces")
    async def get_session_traces(
        session_id: str,
        since: float | None = Query(None),
        limit: int | None = Query(None),
    ):
        traces = await store.get_session_traces(session_id, since=since, limit=limit)
        return traces

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        count = await sessions.delete_session(session_id)
        return {"deleted": count}

    # -- Trace endpoints ---------------------------------------------------

    @app.get("/traces/{trace_id}")
    async def get_trace(trace_id: str):
        trace = await store.get_trace(trace_id)
        if trace is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Trace {trace_id} not found"},
            )
        return trace

    @app.post("/traces/query")
    async def query_traces(request: Request):
        body = await _safe_json(request)
        session_ids = body.get("session_ids", [])
        results: list[dict[str, Any]] = []
        for sid in session_ids:
            traces = await store.get_session_traces(
                sid,
                since=body.get("since"),
                limit=body.get("limit"),
            )
            results.extend(traces)
        return results

    # -- Admin endpoints ---------------------------------------------------

    @app.post("/admin/workers")
    async def add_worker(request: Request):
        body = await _safe_json(request)
        url = body.get("url")
        if not url:
            return JSONResponse(status_code=400, content={"error": "url is required"})
        wid = body.get("worker_id", str(uuid.uuid4()))
        router.add_worker(
            WorkerInfo(
                worker_id=wid,
                url=url,
                model_name=body.get("model_name"),
                weight=body.get("weight", 1),
            )
        )
        # Start health checks if this is the first worker
        if len(router.workers) == 1:
            await router.start_health_checks()
        return {"worker_id": wid, "url": url}

    @app.delete("/admin/workers/{worker_id}")
    async def remove_worker(worker_id: str):
        worker = next((w for w in router.workers if w.worker_id == worker_id), None)
        if worker is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Worker {worker_id} not found"},
            )
        router.remove_worker(worker.url)
        return {"removed": worker_id}

    @app.get("/admin/workers")
    async def list_workers():
        workers = router.get_workers()
        return [w.model_dump() for w in workers]

    @app.post("/admin/flush")
    async def flush():
        await store.flush()
        return {"status": "flushed"}

    @app.post("/admin/reload")
    async def reload():
        # Placeholder for hot-reload
        return {"status": "ok"}

    # -- Proxy catch-all (must be last) ------------------------------------

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    )
    async def proxy_v1(request: Request, path: str):
        # Ensure session exists in manager (implicit creation)
        sid = getattr(request.state, "session_id", None)
        if sid:
            sessions.ensure_session(sid)
        return await proxy.handle(request)

    # Also handle bare /v1 (e.g. /v1/models)
    @app.api_route("/v1", methods=["GET", "POST"])
    async def proxy_v1_root(request: Request):
        sid = getattr(request.state, "session_id", None)
        if sid:
            sessions.ensure_session(sid)
        return await proxy.handle(request)

    # Store references on app for external access
    app.state.config = config  # type: ignore[attr-defined]
    app.state.router = router  # type: ignore[attr-defined]
    app.state.proxy = proxy  # type: ignore[attr-defined]
    app.state.sessions = sessions  # type: ignore[attr-defined]
    app.state.store = store  # type: ignore[attr-defined]

    return app


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _safe_json(request: Request) -> dict[str, Any]:
    try:
        return await request.json()
    except Exception:
        return {}


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------


def _load_config(args: argparse.Namespace) -> GatewayConfig:
    """Build a ``GatewayConfig`` from CLI args, env vars, and optional YAML file."""
    data: dict[str, Any] = {}

    # 1. YAML file (lowest priority)
    config_path = getattr(args, "config", None)
    if config_path:
        with open(config_path) as f:
            data.update(yaml.safe_load(f) or {})

    # 2. Env vars
    env_map = {
        "RLLM_GATEWAY_HOST": "host",
        "RLLM_GATEWAY_PORT": "port",
        "RLLM_GATEWAY_DB_PATH": "db_path",
        "RLLM_GATEWAY_LOG_LEVEL": "log_level",
        "RLLM_GATEWAY_STORE": "store_worker",
    }
    for env_key, config_key in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            if config_key == "port":
                data[config_key] = int(val)
            else:
                data[config_key] = val

    # 3. CLI args (highest priority)
    if getattr(args, "host", None) is not None:
        data["host"] = args.host
    if getattr(args, "port", None) is not None:
        data["port"] = args.port
    if getattr(args, "db_path", None) is not None:
        data["db_path"] = args.db_path
    if getattr(args, "log_level", None) is not None:
        data["log_level"] = args.log_level
    if getattr(args, "store", None) is not None:
        data["store_worker"] = args.store

    # Workers from CLI --worker flags
    worker_urls = getattr(args, "worker", None) or []
    if worker_urls:
        data["workers"] = [{"url": url, "worker_id": str(i)} for i, url in enumerate(worker_urls)]

    return GatewayConfig(**data)


# ------------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="rllm-model-gateway: lightweight LLM call proxy for RL training")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--worker",
        type=str,
        action="append",
        help="Worker URL (can be repeated)",
    )
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--store", type=str, default=None, choices=["sqlite", "memory"])
    parser.add_argument("--log-level", type=str, default=None)

    args = parser.parse_args()
    config = _load_config(args)

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    app = create_app(config)

    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level.lower())


if __name__ == "__main__":
    main()

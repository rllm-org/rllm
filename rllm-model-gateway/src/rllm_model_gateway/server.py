"""FastAPI application factory and CLI entrypoint for rllm-model-gateway."""

import argparse
import asyncio
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
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
# Access-log noise filter
# ------------------------------------------------------------------

# Paths whose access lines are filtered from uvicorn.access. These fire on
# every rollout (often dozens of times per task) and crowd out the lines
# that actually help — chat completions, session create/delete, trace reads.
_NOISY_ACCESS_PATHS: tuple[str, ...] = (
    "/admin/flush",
    "/admin/workers",
    "/health",
    "/health/workers",
)


class _AccessLogPathFilter(logging.Filter):
    """Drop uvicorn.access records whose request path is in _NOISY_ACCESS_PATHS.

    uvicorn.access formats records with positional args:
        (client_addr, method, full_path, http_version, status_code)
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not args or len(args) < 3:
            return True
        path = args[2]
        if not isinstance(path, str):
            return True
        # Strip query string before matching.
        bare = path.split("?", 1)[0]
        return bare not in _NOISY_ACCESS_PATHS


_access_filter_installed = False


def _install_access_log_filter() -> None:
    """Idempotently attach the path filter to the uvicorn.access logger."""
    global _access_filter_installed
    if _access_filter_installed:
        return
    logging.getLogger("uvicorn.access").addFilter(_AccessLogPathFilter())
    _access_filter_installed = True


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
    local_handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
) -> FastAPI:
    """Create and return a fully configured FastAPI application."""
    if config is None:
        config = GatewayConfig()

    _install_access_log_filter()

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
                api_path=wc.api_path,
                model_name=wc.model_name,
                weight=wc.weight,
            )
        )

    # Build the renderer for cumulative token mode. The renderer owns
    # message↔token conversion and the cross-turn bridge (see
    # token_accumulator.TokenAccumulator). The tokenizer is loaded from the
    # served model path (``config.model``), which we assume is a complete,
    # unmodified HuggingFace checkpoint.
    renderer = None
    tokenizer = None
    # use_sglang needs the model tokenizer (to decode /generate completion token
    # ids for tool-call parsing). cumulative_token_mode additionally needs the
    # renderer (the cross-turn bridge that keeps prior sampled tokens verbatim).
    # Non-cumulative use_sglang does NOT need the renderer: input tokenization is
    # done server-side via /tokenize and output parsing via SGLang's parser.
    if config.cumulative_token_mode or config.use_sglang:
        if not config.model:
            raise ValueError("cumulative_token_mode/use_sglang require 'model' to be set in GatewayConfig (path to the served HuggingFace checkpoint).")
        try:
            from transformers import AutoTokenizer
        except ImportError as err:
            raise ImportError("cumulative_token_mode/use_sglang require the 'transformers' package. Install it with: pip install transformers") from err
        tokenizer = AutoTokenizer.from_pretrained(config.model)

    if config.cumulative_token_mode:
        try:
            from renderers import create_renderer
        except ImportError as err:
            raise ImportError("cumulative_token_mode requires the 'renderers' package. Install it with: pip install renderers") from err

        # renderer_family="auto" lets renderers resolve the family by matching the
        # tokenizer's name_or_path against its MODEL_RENDERER_MAP. This succeeds
        # when ``model`` is a canonical HF id (e.g. "Qwen/Qwen3-8B") but misses for
        # a local/custom checkpoint path, which falls back to DefaultRenderer (whose
        # bridge_to_next_turn always returns None, disabling drift protection). When
        # serving from a path, set renderer_family explicitly. Supported families /
        # MODEL_RENDERER_MAP:
        #   https://github.com/PrimeIntellect-ai/renderers/blob/main/renderers/base.py
        renderer = create_renderer(tokenizer, renderer=config.renderer_family)
        logger.info(
            "Built %s (family=%r) from %s for cumulative_token_mode=%s use_sglang=%s",
            type(renderer).__name__,
            config.renderer_family,
            config.model,
            config.cumulative_token_mode,
            config.use_sglang,
        )
        if type(renderer).__name__ == "DefaultRenderer":
            raise ValueError(
                f"Cumulative token mode resolved to DefaultRenderer for renderer_family="
                f"{config.renderer_family!r} (model={config.model!r}). DefaultRenderer "
                "provides no cross-turn bridge, so drift-free token forwarding is disabled. "
                "renderer_family='auto' only resolves when 'model' is a canonical HuggingFace "
                "id present in renderers' MODEL_RENDERER_MAP; a local/custom checkpoint path "
                "will not match. Either pass a recognized HF id as 'model', or set "
                "renderer_family explicitly to match your model (e.g. 'qwen3', 'qwen3.5', "
                "'qwen3.6', 'glm-5', 'deepseek-v3', 'gpt-oss'). Check supported families in "
                "MODEL_RENDERER_MAP of: https://github.com/PrimeIntellect-ai/renderers/blob/"
                "main/renderers/base.py"
            )

    proxy = ReverseProxy(
        router=router,
        store=store,
        strip_vllm=config.strip_vllm_fields,
        sync_traces=config.sync_traces,
        local_handler=local_handler,
        cumulative_token_mode=config.cumulative_token_mode,
        use_sglang=config.use_sglang,
        renderer=renderer,
        tokenizer=tokenizer,
        model=config.model,
        sglang_tool_call_parser=config.sglang_tool_call_parser,
        sglang_reasoning_parser=config.sglang_reasoning_parser,
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
        sessions=sessions,
        model=config.model,
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
            sampling_params=body.get("sampling_params"),
        )
        return {"session_id": sid, "url": f"/sessions/{sid}/v1"}

    @app.get("/sessions")
    async def list_sessions(
        since: float | None = Query(None),
        limit: int | None = Query(None),
    ):
        result = await sessions.list_sessions(since=since, limit=limit)
        return [s.model_dump() for s in result]

    # NOTE: ``{session_id:path}`` allows multi-segment IDs (e.g.
    # ``harbor/hello-world:0`` from namespaced Harbor tasks). FastAPI/
    # Starlette match routes in declaration order, so the more-specific
    # ``/sessions/{session_id:path}/traces`` MUST be declared before the
    # bare ``/sessions/{session_id:path}`` — otherwise the bare route's
    # greedy capture swallows ``/sessions/foo/traces`` as
    # ``session_id="foo/traces"`` and the traces endpoint is unreachable.

    @app.get("/sessions/{session_id:path}/traces")
    async def get_session_traces(
        session_id: str,
        since: float | None = Query(None),
        limit: int | None = Query(None),
    ):
        traces = await store.get_session_traces(session_id, since=since, limit=limit)
        return traces

    @app.get("/sessions/{session_id:path}")
    async def get_session(session_id: str):
        info = await sessions.get_session_info(session_id)
        if info is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Session {session_id} not found"},
            )
        return info.model_dump()

    @app.delete("/sessions/{session_id:path}")
    async def delete_session(session_id: str):
        proxy._accumulators.pop(session_id, None)
        count = await sessions.delete_session(session_id)
        return {"deleted": count}

    @app.post("/sessions/batch_delete")
    async def batch_delete_sessions(request: Request):
        body = await _safe_json(request)
        session_ids = body.get("session_ids", [])
        total = 0
        for sid in session_ids:
            proxy._accumulators.pop(sid, None)
            total += await sessions.delete_session(sid)
        return {"deleted": total}

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
        # Build kwargs — omit api_path if not provided so the validator auto-splits
        worker_kwargs: dict[str, Any] = {
            "worker_id": wid,
            "url": url,
            "model_name": body.get("model_name"),
            "weight": body.get("weight", 1),
        }
        if "api_path" in body:
            worker_kwargs["api_path"] = body["api_path"]
        worker = WorkerInfo(**worker_kwargs)
        router.add_worker(worker)
        # Start health checks if this is the first worker
        if len(router.workers) == 1:
            await router.start_health_checks()
        return {"worker_id": wid, "url": worker.url, "api_path": worker.api_path}

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
        # Drain in-flight fire-and-forget _safe_store tasks queued by the
        # non-streaming and streaming paths in Proxy._persist. Without this,
        # /admin/flush only flushes the store's own buffers and a trace
        # captured milliseconds before this call may not yet be persisted —
        # the trainer's aget_traces would then return N-1 traces for an
        # agent that made N calls, causing positional mismatch downstream.
        pending = list(proxy._pending_traces)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await store.flush()
        return {"status": "flushed"}

    @app.post("/admin/reload")
    async def reload():
        # Placeholder for hot-reload
        return {"status": "ok"}

    @app.get("/admin/weight_version")
    async def get_weight_version():
        return {"weight_version": proxy.weight_version}

    @app.post("/admin/weight_version")
    async def set_weight_version(request: Request):
        body = await _safe_json(request)
        version = body.get("weight_version")
        if version is None:
            return JSONResponse(status_code=400, content={"error": "weight_version is required"})
        try:
            proxy.weight_version = int(version)
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content={"error": f"invalid weight_version: {version!r}"})
        return {"weight_version": proxy.weight_version}

    # -- Proxy catch-all (must be last) ------------------------------------

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST"],
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
    if getattr(args, "model", None) is not None:
        data["model"] = args.model
    if getattr(args, "cumulative_token_mode", False):
        data["cumulative_token_mode"] = True
    if getattr(args, "use_sglang", False):
        data["use_sglang"] = True
    if getattr(args, "renderer_family", None) is not None:
        data["renderer_family"] = args.renderer_family
    if getattr(args, "sglang_tool_call_parser", None) is not None:
        data["sglang_tool_call_parser"] = args.sglang_tool_call_parser
    if getattr(args, "sglang_reasoning_parser", None) is not None:
        data["sglang_reasoning_parser"] = args.sglang_reasoning_parser

    # Workers from CLI --worker flags (WorkerConfig validator auto-splits URLs)
    worker_urls = getattr(args, "worker", None) or []
    if worker_urls:
        data["workers"] = [{"url": raw_url, "worker_id": str(i)} for i, raw_url in enumerate(worker_urls)]

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
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="If set, the gateway rewrites every request body's 'model' field to this value before forwarding.",
    )
    parser.add_argument(
        "--cumulative-token-mode",
        action="store_true",
        default=False,
        help="Enable cumulative token mode for drift-free multi-turn RL training. Loads the tokenizer from --model (the served HuggingFace checkpoint).",
    )
    parser.add_argument(
        "--use-sglang",
        action="store_true",
        default=False,
        help="Route generation through SGLang's native /generate API (token ids + "
        "logprobs in meta_info, works through sgl-router) instead of the OpenAI "
        "/v1/{chat/,}completions endpoints. Requires --model and a renderer; the "
        "gateway renders each turn's prompt to token ids itself. Required for RL "
        "with SGLang server or sgl-router as inference endpoint.",
    )
    parser.add_argument(
        "--sglang-tool-call-parser",
        type=str,
        default=None,
        help="SGLang function-call parser name (e.g. 'qwen', 'llama3', 'deepseekv3') used "
        "in use_sglang mode to parse tool calls from /generate output text, required for"
        "tool-using agents in use_sglang mode.",
    )
    parser.add_argument(
        "--sglang-reasoning-parser",
        type=str,
        default=None,
        help="SGLang reasoning parser name (e.g. 'qwen3', 'deepseek-r1') to split <think>...</think> reasoning from output text in use_sglang mode. Optional.",
    )
    parser.add_argument(
        "--renderer-family",
        type=str,
        default=None,
        help="renderers family for the cumulative-mode bridge (e.g. 'qwen3', 'qwen3.5', "
        "'qwen3.6', 'glm-5', 'deepseek-v3', 'gpt-oss'). Renderers can auto infer it if --model "
        "is a huggingface model id, but if --model is a local path, you must explicitly set it. "
        "Check the supported model families in MODEL_RENDERER_MAP of "
        "https://github.com/PrimeIntellect-ai/renderers/blob/main/renderers/base.py",
    )

    args = parser.parse_args()
    config = _load_config(args)

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    app = create_app(config)

    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level.lower())


if __name__ == "__main__":
    main()

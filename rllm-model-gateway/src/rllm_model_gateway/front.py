"""Thin session-sharding reverse proxy ("front") for multi-worker gateways.

Runs in front of N backend gateway workers (each an ordinary
``num_workers=1`` gateway) and routes by ``session_id``:

- ``/sessions/{sid}/...`` (chat, traces, get, delete) → the worker that owns
  ``{sid}`` (deterministic ``ConsistentHashPolicy`` over the fixed worker set).
  The **original path is forwarded unchanged**, so the worker's own middleware
  keys its per-session state — the whole reason the front is a pass-through and
  not the gateway app in proxy mode.
- ``POST /sessions`` (create): ``{sid}`` is in the body → route to its owner.
- global ops (``POST /admin/weight_version|flush|reload``,
  ``POST /sessions/batch_delete``) → fan out to all workers.
- ``/health`` → ok.

Responses are **streamed** through (preserves SSE + the whitespace heartbeat).
The front never parses chat bodies or touches session state — the workers do
all of that. It is deliberately thin so one core routes for many workers; run
several via SO_REUSEPORT if it ever saturates.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from rllm_model_gateway import fastjson
from rllm_model_gateway.models import WorkerInfo
from rllm_model_gateway.session_router import ConsistentHashPolicy

logger = logging.getLogger(__name__)

# Global ops that must reach every worker (workers hold their own shard of state
# / build their own traces), not a single session's owner.
_GLOBAL_POST_PATHS = frozenset({"/admin/weight_version", "/admin/flush", "/admin/reload", "/sessions/batch_delete"})

_DROP_REQ_HEADERS = frozenset({"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade"})
_DROP_RESP_HEADERS = frozenset({"content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade"})


def _extract_session_id(path: str) -> str | None:
    """Extract ``{sid}`` from ``/sessions/{sid}/v1/...``, ``/sessions/{sid}/traces``,
    or ``/sessions/{sid}``. ``{sid}`` may be multi-segment (e.g. ``harbor/task:0``)."""
    prefix = "/sessions/"
    if not path.startswith(prefix):
        return None
    rest = path[len(prefix) :]
    for suffix in ("/v1", "/traces"):
        i = rest.find(suffix + "/")
        if i != -1:
            return rest[:i]
        if rest.endswith(suffix):
            return rest[: -len(suffix)]
    return rest or None


class Front:
    def __init__(self, worker_urls: list[str]) -> None:
        if not worker_urls:
            raise ValueError("front requires at least one --worker URL")
        self.workers = [WorkerInfo(worker_id=str(i), url=u) for i, u in enumerate(worker_urls)]
        self.policy = ConsistentHashPolicy()
        self.policy.on_worker_change(self.workers)
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=None),  # workers own the timeouts / heartbeat
            follow_redirects=True,
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200),
        )

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _owner(self, session_id: str) -> WorkerInfo:
        return self.policy.select_worker(self.workers, session_id, {})

    def _req_headers(self, request: Request) -> dict[str, str]:
        return {k: v for k, v in request.headers.items() if k.lower() not in _DROP_REQ_HEADERS}

    def _resp_headers(self, resp: httpx.Response) -> dict[str, str]:
        return {k: v for k, v in resp.headers.items() if k.lower() not in _DROP_RESP_HEADERS}

    async def handle(self, request: Request) -> Response:
        path = request.url.path
        method = request.method
        if path == "/health":
            return JSONResponse({"status": "ok", "workers": len(self.workers)})

        body = await request.body()

        if method == "POST" and path in _GLOBAL_POST_PATHS:
            return await self._fan_out(method, path, request, body)

        if method == "POST" and path == "/sessions":  # create: sid in body
            sid = None
            try:
                sid = fastjson.loads(body).get("session_id")
            except Exception:  # noqa: BLE001
                pass
            target = self._owner(sid) if sid else self.workers[0]
            return await self._forward(target, request, body, stream=False)

        sid = _extract_session_id(path)
        if sid is not None:
            # Stream so SSE + the worker's whitespace heartbeat pass through.
            return await self._forward(self._owner(sid), request, body, stream=True)

        # Unrecognized (e.g. GET /sessions list, /traces/{id}) — best-effort to worker 0.
        return await self._forward(self.workers[0], request, body, stream=False)

    async def _forward(self, worker: WorkerInfo, request: Request, body: bytes, *, stream: bool) -> Response:
        assert self._client is not None
        url = f"{worker.url}{request.url.path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        headers = self._req_headers(request)

        if not stream:
            resp = await self._client.request(request.method, url, headers=headers, content=body or None)
            return Response(content=resp.content, status_code=resp.status_code, headers=self._resp_headers(resp))

        req = self._client.build_request(request.method, url, headers=headers, content=body or None)
        resp = await self._client.send(req, stream=True)

        async def _body_iter():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(_body_iter(), status_code=resp.status_code, headers=self._resp_headers(resp))

    async def _fan_out(self, method: str, path: str, request: Request, body: bytes) -> Response:
        assert self._client is not None
        headers = self._req_headers(request)

        async def _one(w: WorkerInfo):
            try:
                return await self._client.request(method, f"{w.url}{path}", headers=headers, content=body or None)
            except Exception as e:  # noqa: BLE001
                return e

        results = await asyncio.gather(*[_one(w) for w in self.workers])
        merged: dict = {"workers": len(self.workers), "ok": 0, "errors": 0, "deleted": 0}
        weight_version = None
        for r in results:
            if isinstance(r, Exception):
                merged["errors"] += 1
                logger.warning("front fan-out to a worker failed for %s: %s", path, r)
                continue
            merged["ok"] += 1
            try:
                j = r.json()
                if isinstance(j.get("deleted"), int):
                    merged["deleted"] += j["deleted"]
                if "weight_version" in j:
                    weight_version = j["weight_version"]
            except Exception:  # noqa: BLE001
                pass
        if weight_version is not None:
            merged["weight_version"] = weight_version
        return JSONResponse(merged)


def create_front_app(worker_urls: list[str]) -> Starlette:
    front = Front(worker_urls)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        await front.start()
        logger.info("Gateway front started over %d workers: %s", len(front.workers), [w.url for w in front.workers])
        yield
        await front.stop()

    async def _catch_all(request: Request) -> Response:
        return await front.handle(request)

    routes = [Route("/{path:path}", _catch_all, methods=["GET", "POST", "PUT", "DELETE", "PATCH"])]
    return Starlette(routes=routes, lifespan=lifespan)

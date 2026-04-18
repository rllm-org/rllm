"""Two-tier API key auth.

Token sources accepted (in order):
    Authorization: Bearer <key>          # OpenAI SDK convention
    x-api-key: <key>                     # Anthropic SDK convention

Endpoint classification:
    /health                               — public, no auth
    /sessions/{sid}/v1/...                — accepts agent OR admin key
    everything else                       — admin key only
"""

from __future__ import annotations

import hmac
import json

from starlette.types import ASGIApp, Receive, Scope, Send

from rllm_model_gateway.middleware import SESSION_PATH_RE


class AuthMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        admin_api_key: str,
        agent_api_key: str,
    ) -> None:
        self.app = app
        self.admin_api_key = admin_api_key
        self.agent_api_key = agent_api_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope["path"]
        if path == "/health":
            await self.app(scope, receive, send)
            return

        token = _extract_bearer(scope)
        is_proxy = bool(SESSION_PATH_RE.search(path))

        if token is None:
            await _send_401(send)
            return

        # Constant-time comparison to avoid leaking key length / mismatch
        # position via response-time differences.
        if is_proxy:
            ok = _safe_eq(token, self.admin_api_key) or _safe_eq(token, self.agent_api_key)
        else:
            ok = _safe_eq(token, self.admin_api_key)

        if not ok:
            await _send_401(send)
            return

        await self.app(scope, receive, send)


def _safe_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def _extract_bearer(scope: Scope) -> str | None:
    for name, value in scope.get("headers", []):
        n = name.decode("latin-1").lower() if isinstance(name, bytes) else name.lower()
        v = value.decode("latin-1") if isinstance(value, bytes) else value
        if n == "authorization":
            if v.lower().startswith("bearer "):
                return v[7:].strip()
        elif n == "x-api-key":
            return v.strip()
    return None


async def _send_401(send: Send) -> None:
    body = json.dumps({"error": "Unauthorized"}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})

from __future__ import annotations

import re

from starlette.types import ASGIApp, Receive, Scope, Send

# Matches /sessions/{sid}/v1/<endpoint-tail>. Excludes management endpoints
# like /sessions/{sid}/traces by requiring /v1 to follow.
# Used by both SessionRoutingMiddleware (here) and AuthMiddleware (auth.py).
SESSION_PATH_RE = re.compile(r"/sessions/([^/]+)(/v1(?:/.*)?)$")


class SessionRoutingMiddleware:
    """Extract session_id from /sessions/{sid}/v1/... and rewrite the path.

    Downstream handlers read ``scope['state']['session_id']``.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope["path"]
        session_id: str | None = None

        m = SESSION_PATH_RE.search(path)
        if m:
            session_id = m.group(1)
            path = m.group(2)

        state = scope.setdefault("state", {})
        state["session_id"] = session_id
        scope["path"] = path
        if "raw_path" in scope:
            scope["raw_path"] = path.encode("utf-8")

        await self.app(scope, receive, send)

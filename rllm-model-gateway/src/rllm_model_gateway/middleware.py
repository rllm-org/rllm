"""SessionRoutingMiddleware — extracts session ID from URL and injects sampling params.

Handles the ``/sessions/{sid}/v1/...`` URL pattern (inspired by miles router).

Injects ``logprobs=True`` and ``return_token_ids=True`` (when configured)
into the request body before forwarding.
"""

import json
import logging
import re
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

_SESSION_PATH_RE = re.compile(r"/sessions/([^/]+)(/v1(?:/.*)?)$")


class SessionRoutingMiddleware:
    """Pure-ASGI middleware that rewrites paths and injects sampling parameters.

    After this middleware runs, downstream handlers can read:
    - ``scope["state"]["session_id"]`` — the extracted session ID (or ``None``)

    The URL path is rewritten to strip the ``/sessions/{sid}`` prefix so that
    downstream route matching sees ``/v1/chat/completions``, etc.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        add_logprobs: bool = True,
        add_return_token_ids: bool = True,
    ) -> None:
        self.app = app
        self.add_logprobs = add_logprobs
        self.add_return_token_ids = add_return_token_ids

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope["path"]
        session_id: str | None = None

        # Extract session_id from /sessions/{sid}/v1/...
        m = _SESSION_PATH_RE.search(path)
        if m:
            session_id = m.group(1)
            path = m.group(2)  # already starts with /v1

        # Store extracted data in scope state
        state = scope.setdefault("state", {})
        state["session_id"] = session_id

        # Rewrite path
        scope["path"] = path
        # Also update raw_path if present
        if "raw_path" in scope:
            scope["raw_path"] = path.encode("utf-8")

        # For methods that carry a body, inject sampling parameters
        method = scope.get("method", "").upper()
        if method in ("POST", "PUT", "PATCH") and (self.add_logprobs or self.add_return_token_ids):
            await self._inject_params(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    async def _inject_params(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Read body, inject sampling params, then forward with mutated body."""
        body_parts: list[bytes] = []
        more = True
        while more:
            msg = await receive()
            body_parts.append(msg.get("body", b""))
            more = msg.get("more_body", False)

        raw = b"".join(body_parts)
        if raw:
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    # Record whether the client originally requested logprobs
                    # so the proxy can strip them from the response if not.
                    state = scope["state"]
                    state["originally_requested_logprobs"] = "logprobs" in payload and payload["logprobs"]
                    self._mutate(payload)
                    raw = json.dumps(payload).encode("utf-8")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # non-JSON body — forward as-is

        # Build a receive that replays the (possibly mutated) body once,
        # then delegates to the original receive for disconnect detection.
        # This is critical: Starlette's StreamingResponse concurrently
        # listens for client disconnect via receive().  If we return
        # http.disconnect immediately, it aborts the streaming response.
        body_sent = False

        async def patched_receive() -> Message:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": raw, "more_body": False}
            # Delegate to original receive for disconnect detection —
            # it will block until the response is complete.
            return await receive()

        await self.app(scope, patched_receive, send)

    def _mutate(self, payload: dict[str, Any]) -> None:
        """Inject ``logprobs`` and ``return_token_ids`` into the request body."""
        if self.add_logprobs and "logprobs" not in payload:
            payload["logprobs"] = True
        if self.add_return_token_ids:
            # extra_body is the OpenAI Python SDK convention for vendor extensions
            extra = payload.setdefault("extra_body", {})
            if isinstance(extra, dict) and "return_token_ids" not in extra:
                extra["return_token_ids"] = True
            # Also set at root for direct vLLM calls
            if "return_token_ids" not in payload:
                payload["return_token_ids"] = True

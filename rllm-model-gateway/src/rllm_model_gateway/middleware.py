"""SessionRoutingMiddleware — extracts session ID from URL and injects sampling params.

Handles the ``/sessions/{sid}/v1/...`` URL pattern (inspired by miles router).

Injects ``logprobs=True`` and ``return_token_ids=True`` (when configured)
into the request body before forwarding.

Also exposes ``_read_body`` / ``_body_replay`` — ASGI body helpers reused by
``responses_middleware.ResponsesAdapterMiddleware``.
"""

import json
import logging
import re
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

# ``.+?`` (non-greedy) so multi-segment session IDs work — e.g.
# ``harbor/hello-world:0`` from a namespaced Harbor task. The ``/v1``
# suffix anchor forces the shortest match that still leaves a valid
# ``/v1[/...]`` tail, so legacy single-segment ids (no slash) capture
# identically to the old ``[^/]+`` pattern.
_SESSION_PATH_RE = re.compile(r"/sessions/(.+?)(/v1(?:/.*)?)$")


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
        sessions: Any | None = None,
        model: str | None = None,
    ) -> None:
        self.app = app
        self.add_logprobs = add_logprobs
        self.add_return_token_ids = add_return_token_ids
        self.sessions = sessions  # SessionManager — for per-session sampling params
        self.model = model

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

        # Inject sampling parameters into POST request bodies (chat completions, etc.)
        method = scope.get("method", "").upper()
        needs_injection = self.add_logprobs or self.add_return_token_ids or self.sessions is not None
        if method == "POST" and needs_injection:
            await self._inject_params(scope, receive, send, session_id)
        else:
            await self.app(scope, receive, send)

    async def _inject_params(self, scope: Scope, receive: Receive, send: Send, session_id: str | None = None) -> None:
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
                    # Guard: an outer middleware may have already recorded this
                    # from a different request convention (e.g. Responses API's
                    # ``include`` list). Only fall back to the Chat convention
                    # when nobody upstream has set it.
                    state = scope["state"]
                    if "originally_requested_logprobs" not in state:
                        state["originally_requested_logprobs"] = "logprobs" in payload and payload["logprobs"]
                    self._mutate(payload, session_id)
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

    def _mutate(self, payload: dict[str, Any], session_id: str | None = None) -> None:
        """Inject logprobs / return_token_ids / model pin, then overwrite the session's sampling params.

        Keys in the session config overwrite whatever the client sent; keys absent
        from it pass through untouched.
        """
        if self.add_logprobs and "logprobs" not in payload:
            payload["logprobs"] = True
        if self.add_return_token_ids and "return_token_ids" not in payload:
            payload["return_token_ids"] = True
        # Pin the model the gateway forwards to (overrides whatever the client sets)
        if self.model:
            payload["model"] = self.model
        if session_id and self.sessions is not None:
            sp = self.sessions.get_sampling_params(session_id)
            if sp:
                payload.update(sp)


# ---------------------------------------------------------------------------
# Shared ASGI body helpers (also usable by future middleware)
# ---------------------------------------------------------------------------


async def _read_body(receive: Receive) -> bytes:
    """Read all HTTP body frames from an ASGI ``receive`` into a single bytes."""
    parts: list[bytes] = []
    while True:
        msg = await receive()
        parts.append(msg.get("body", b""))
        if not msg.get("more_body", False):
            break
    return b"".join(parts)


def _body_replay(raw: bytes, original_receive: Receive) -> Receive:
    """Wrap ``original_receive`` so it emits ``raw`` once, then delegates.

    Downstream ASGI apps read the body via ``receive()``; if we consumed it
    ourselves we must synthesize a single ``http.request`` frame with the
    (possibly rewritten) body, then hand back the original receive for
    disconnect detection.
    """
    sent = False

    async def _receive() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": raw, "more_body": False}
        return await original_receive()

    return _receive

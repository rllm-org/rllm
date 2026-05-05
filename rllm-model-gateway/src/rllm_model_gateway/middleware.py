"""SessionRoutingMiddleware — extracts rLLM session/harness metadata and
injects sampling params.

Three sources of session identity, in precedence order:

1. ``X-RLLM-*`` headers (canonical convention; harness shims stamp these).
2. Body fallback: ``request.metadata.rllm`` or top-level ``request.rllm``.
3. Legacy URL path: ``/sessions/{sid}/v1/...`` — kept for back-compat with
   the existing training proxy entry points.

After this middleware runs, downstream handlers can read:
- ``scope["state"]["session_id"]`` — the extracted session ID (or ``None``).
- ``scope["state"]["rllm_metadata"]`` — full ``RllmMetadata`` (run_id,
  harness, step_id, parent_span_id, project, experiment).
- ``scope["state"]["originally_requested_logprobs"]`` — used by the proxy
  to strip injected logprobs from the response.

The URL path is rewritten to strip the ``/sessions/{sid}`` prefix so that
downstream route matching sees ``/v1/chat/completions``, etc.
"""

import json
import logging
import re
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rllm_model_gateway.metadata import (
    RllmMetadata,
    extract_metadata,
    headers_from_scope,
)

logger = logging.getLogger(__name__)

_SESSION_PATH_RE = re.compile(r"/sessions/([^/]+)(/v1(?:/.*)?)$")

_BEARER_PREFIX = "bearer "


async def _send_401(send: Send) -> None:
    """Send a minimal 401 Unauthorized so callers get a clean failure mode."""
    body = b'{"error": {"message": "missing or invalid bearer token", "type": "auth_error"}}'
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b'Bearer realm="rllm-gateway"'),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


class SessionRoutingMiddleware:
    """Pure-ASGI middleware that rewrites paths and injects sampling parameters."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        add_logprobs: bool = True,
        add_return_token_ids: bool = True,
        sessions: Any | None = None,
        sampling_params_priority: str = "client",
        model: str | None = None,
        inbound_auth_token: str | None = None,
    ) -> None:
        if sampling_params_priority not in ("client", "session"):
            raise ValueError(f"sampling_params_priority must be 'client' or 'session', got {sampling_params_priority!r}")
        self.app = app
        self.add_logprobs = add_logprobs
        self.add_return_token_ids = add_return_token_ids
        self.sessions = sessions  # SessionManager — for per-session sampling params
        self.sampling_params_priority = sampling_params_priority
        self.model = model
        # When set, every inbound HTTP request must carry
        # ``Authorization: Bearer <inbound_auth_token>``. Used by the
        # eval gateway when exposed via a public tunnel — without this,
        # anyone who guesses the tunnel URL can burn provider credits.
        self.inbound_auth_token = inbound_auth_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = headers_from_scope(scope)
        original_path: str = scope["path"]

        # Inbound auth check — runs before any body parsing or routing.
        if self.inbound_auth_token is not None and not self._auth_ok(headers):
            await _send_401(send)
            return

        # Strip /sessions/{sid}/v1 prefix so downstream routes see /v1/...
        new_path = original_path
        m = _SESSION_PATH_RE.search(original_path)
        if m:
            new_path = m.group(2)
        scope["path"] = new_path
        if "raw_path" in scope:
            scope["raw_path"] = new_path.encode("utf-8")

        state = scope.setdefault("state", {})

        method = scope.get("method", "").upper()
        needs_body_inspection = method == "POST" and (self.add_logprobs or self.add_return_token_ids or self.sessions is not None)

        if needs_body_inspection:
            await self._inject_params(scope, receive, send, headers=headers, original_path=original_path)
        else:
            metadata = extract_metadata(headers=headers, path=original_path)
            self._populate_state(state, metadata)
            await self.app(scope, receive, send)

    def _auth_ok(self, headers: dict[str, str]) -> bool:
        """Constant-time check that the request carries our bearer token."""
        import hmac

        auth = headers.get("authorization", "")
        if not auth.lower().startswith(_BEARER_PREFIX):
            return False
        presented = auth[len(_BEARER_PREFIX) :].strip()
        # ``compare_digest`` to avoid leaking the token via timing.
        return hmac.compare_digest(presented, self.inbound_auth_token or "")

    @staticmethod
    def _populate_state(state: dict[str, Any], metadata: RllmMetadata) -> None:
        state["rllm_metadata"] = metadata
        state["session_id"] = metadata.session_id

    async def _inject_params(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        headers: dict[str, str],
        original_path: str,
    ) -> None:
        """Read body, extract metadata + inject sampling params, then forward with mutated body."""
        body_parts: list[bytes] = []
        more = True
        while more:
            msg = await receive()
            body_parts.append(msg.get("body", b""))
            more = msg.get("more_body", False)

        raw = b"".join(body_parts)
        payload: dict[str, Any] = {}
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    payload = parsed
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = {}

        metadata = extract_metadata(headers=headers, body=payload, path=original_path)
        state = scope["state"]
        self._populate_state(state, metadata)

        if payload:
            state["originally_requested_logprobs"] = "logprobs" in payload and payload["logprobs"]
            self._mutate(payload, metadata.session_id)
            raw = json.dumps(payload).encode("utf-8")

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
        """Inject sampling params and strip rLLM scaffolding from the body."""
        if self.add_logprobs and "logprobs" not in payload:
            payload["logprobs"] = True
        if self.add_return_token_ids and "return_token_ids" not in payload:
            payload["return_token_ids"] = True
        # Pin the model the gateway forwards to (overrides whatever the client sets)
        if self.model:
            payload["model"] = self.model
        # Inject per-session sampling params using the configured priority.
        if session_id and self.sessions is not None:
            sp = self.sessions.get_sampling_params(session_id)
            if sp:
                if self.sampling_params_priority == "session":
                    payload.update(sp)  # session wins on conflict
                else:  # "client"
                    for key, value in sp.items():
                        if key not in payload:
                            payload[key] = value
        # Strip rLLM-specific body fields so providers never see them.
        payload.pop("rllm", None)
        md = payload.get("metadata")
        if isinstance(md, dict):
            md.pop("rllm", None)
            if not md:
                payload.pop("metadata", None)

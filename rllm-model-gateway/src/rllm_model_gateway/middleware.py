"""SessionRoutingMiddleware — extracts session ID from URL and injects sampling params.

Handles the ``/sessions/{sid}/v1/...`` URL pattern (inspired by miles router).

Injects ``logprobs=True`` and ``return_token_ids=True`` (when configured)
into the request body before forwarding.

Also provides ``ResponsesAdapterMiddleware`` which translates OpenAI Responses
API requests/responses to/from Chat Completions before ``SessionRoutingMiddleware``
sees them, so downstream (data extraction, cumulative TITO) can assume Chat format.
"""

import json
import logging
import re
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rllm_model_gateway._responses_compat import ResponsesAdapter

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
                    state = scope["state"]
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


# ---------------------------------------------------------------------------
# ResponsesAdapterMiddleware — Responses <-> Chat Completions translation
# ---------------------------------------------------------------------------

# Non-greedy ``.+?`` so the session prefix (which may contain slashes) is
# captured shortest. ``^...$`` fullmatch keeps this from matching sub-paths
# like ``/foo/v1/responses/bar`` — only exact ``/v1/responses`` endpoints.
_RESP_PATH_RE = re.compile(r"^(/sessions/.+?)?/v1/responses$")


class ResponsesAdapterMiddleware:
    """Translate ``/v1/responses`` ↔ ``/v1/chat/completions``.

    Request:  Responses API → Chat Completions, path is rewritten so downstream
              routes see ``/v1/chat/completions``.
    Response: Chat Completions → Responses API (non-streaming buffers and
              retranslates; streaming uses ``_SSETranslatingSend`` per event).

    The ``session_manager`` kwarg is required so ``input_image`` data URLs
    seen this turn can be pushed into session state via
    ``session_manager.set_images(sid, urls)`` for the cumulative-token proxy
    to consume via ``vlm_tito.apply_vlm_tito``.

    Codex CLI's Responses API is cumulative — each turn's body carries the
    full input history including prior ``input_image`` blocks — so we ``set``
    (overwrite), not append, to avoid duplicate accumulation.
    """

    def __init__(self, app: ASGIApp, *, session_manager: Any) -> None:
        # session_manager is a SessionManager instance; typed as Any to avoid
        # a circular import (session_manager imports nothing from here, but a
        # future change might). RLLM-STRICT: middleware refuses to serve if
        # session_manager is None, since a live translation without SM would
        # silently drop image state.
        if session_manager is None:
            raise ValueError("ResponsesAdapterMiddleware requires a session_manager")
        self.app = app
        self.session_manager = session_manager
        self.adapter = ResponsesAdapter()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _RESP_PATH_RE.match(scope["path"]):
            await self.app(scope, receive, send)
            return

        raw = await _read_body(receive)
        result = self._translate_request(raw, scope)
        if result is None:
            # Body not JSON, or empty: pass through untranslated. The replayed
            # receive delivers the original raw bytes to the downstream app.
            await self.app(scope, _body_replay(raw, receive), send)
            return

        chat_raw, ctx = result
        replay = _body_replay(chat_raw, receive)
        if ctx.get("is_stream"):
            await self.app(scope, replay, _SSETranslatingSend(send, ctx, self.adapter))
        else:
            await self._handle_non_streaming(scope, replay, send, ctx)

    def _translate_request(self, raw: bytes, scope: Scope) -> tuple[bytes, dict[str, Any]] | None:
        """Parse Responses body, translate to Chat, push images to SessionManager.

        Returns None if the body is not JSON (caller should pass through).
        On success: rewrites ``scope["path"]`` from ``/v1/responses`` to
        ``/v1/chat/completions`` (preserving any ``/sessions/{sid}`` prefix)
        and stores the adapter ctx on scope state for later response translation.
        """
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict):
            return None

        is_stream = bool(payload.get("stream", False))
        chat_body, ctx = self.adapter.to_chat_completion(payload)
        ctx["is_stream"] = is_stream

        # Session image bookkeeping. SessionRoutingMiddleware strips
        # ``/sessions/{sid}`` before its own path rewrite, but we run BEFORE
        # it (registered later == wrapped outer), so we still see the raw
        # incoming path here.
        m = _RESP_PATH_RE.match(scope["path"])
        sid_seg = m.group(1) if m else None  # ``/sessions/{sid}`` or None
        images = ctx.get("images") or []
        if sid_seg and images:
            sid = sid_seg[len("/sessions/"):]
            self.session_manager.set_images(sid, images)

        # Rewrite path: /v1/responses -> /v1/chat/completions, preserving prefix
        path: str = scope["path"]
        rewritten = path[: -len("/v1/responses")] + "/v1/chat/completions"
        scope["path"] = rewritten
        if "raw_path" in scope:
            scope["raw_path"] = rewritten.encode("utf-8")

        return json.dumps(chat_body).encode("utf-8"), ctx

    async def _handle_non_streaming(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        ctx: dict[str, Any],
    ) -> None:
        """Buffer the downstream response body, translate, and forward."""
        held_start: list[Message] = []
        body_parts: list[bytes] = []

        async def capturing_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                held_start.append(message)
                return
            if message["type"] != "http.response.body":
                await send(message)
                return

            body_parts.append(message.get("body", b""))
            if message.get("more_body", False):
                return

            raw = b"".join(body_parts)
            if raw:
                try:
                    chat_resp = json.loads(raw)
                    body = json.dumps(self.adapter.from_chat_completion(chat_resp, ctx)).encode("utf-8")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    body = raw
            else:
                body = raw
            if held_start:
                start = held_start[0]
                MutableHeaders(scope=start)["content-length"] = str(len(body))
                await send(start)
            await send({"type": "http.response.body", "body": body, "more_body": False})

        await self.app(scope, receive, capturing_send)


class _SSETranslatingSend:
    """ASGI send wrapper: buffer SSE events at ``\\n\\n``, translate per event.

    Codex expects Responses API events (``event: response.xxx\\ndata: {...}\\n\\n``).
    The downstream proxy emits Chat Completions SSE (``data: {chunk}\\n\\n``).
    We buffer bytes as they arrive; whenever a full event boundary appears,
    parse the ``data: {...}`` line, ask the adapter to translate it into
    zero-or-more Responses events, and forward those. On ``more_body=False``
    we flush any trailing buffer and emit terminal ``response.completed`` events.
    """

    def __init__(self, send: Send, ctx: dict[str, Any], adapter: Any) -> None:
        self._send = send
        self._ctx = ctx
        self._adapter = adapter
        self._buffer = b""
        # Media type on start message: keep as-is (text/event-stream). We do
        # not know content-length up front (streaming), so no header rewrite.

    async def __call__(self, message: Message) -> None:
        if message["type"] != "http.response.body":
            await self._send(message)
            return

        self._buffer += message.get("body", b"")
        more_body = message.get("more_body", False)

        # Drain complete events (delimited by blank line \n\n)
        while b"\n\n" in self._buffer:
            event_bytes, self._buffer = self._buffer.split(b"\n\n", 1)
            translated = self._translate_event(event_bytes)
            if translated:
                await self._send(
                    {"type": "http.response.body", "body": translated.encode("utf-8"), "more_body": True}
                )

        if not more_body:
            # Emit any trailing partial event (best-effort), then completion events.
            tail = ""
            if self._buffer.strip():
                translated_tail = self._translate_event(self._buffer)
                if translated_tail:
                    tail += translated_tail
            for line in self._adapter.flush_stream(self._ctx):
                tail += line
            self._buffer = b""
            await self._send(
                {"type": "http.response.body", "body": tail.encode("utf-8"), "more_body": False}
            )

    def _translate_event(self, event_bytes: bytes) -> str:
        """Translate a single SSE event's bytes into concatenated Responses events."""
        text = event_bytes.decode("utf-8", errors="replace").strip()
        if not text.startswith("data: "):
            # Non-data lines (comments, event: types w/o data) — pass through.
            return text + "\n\n" if text else ""
        data_str = text[6:].strip()
        if data_str == "[DONE]":
            # Chat's [DONE] terminates the stream. Suppress here — flush_stream
            # emits response.completed which is the Responses-side equivalent.
            return ""
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            return text + "\n\n"
        events = self._adapter.translate_stream_chunk(chunk, self._ctx)
        return "".join(events)

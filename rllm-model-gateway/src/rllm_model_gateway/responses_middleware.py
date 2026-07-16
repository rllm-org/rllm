"""ResponsesAdapterMiddleware — translate OpenAI Responses API ↔ Chat Completions.

Runs OUTSIDE ``SessionRoutingMiddleware`` (registered later == wrapped outer,
executed first) so that by the time downstream sees the request, the body is
Chat-shaped and the path is ``/v1/chat/completions``. In the reverse direction
we buffer non-streaming responses and translate SSE events on the fly for
streaming.

Records the client's ``include: [message.output_text.logprobs]`` intent onto
``scope.state["originally_requested_logprobs"]`` before delegating downstream,
where ``SessionRoutingMiddleware`` sees the guard and doesn't overwrite.
"""

import json
import re
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rllm_model_gateway._responses_compat import ResponsesAdapter
from rllm_model_gateway.middleware import _body_replay, _read_body


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

        # Record client intent (Responses ``include`` convention) into scope
        # state BEFORE downstream SessionRoutingMiddleware runs. That middleware
        # falls back to the Chat ``logprobs`` field only if we don't set this
        # first, so each middleware stays scoped to its own protocol convention.
        state = scope.setdefault("state", {})
        state["originally_requested_logprobs"] = (
            "message.output_text.logprobs" in (payload.get("include") or [])
        )

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

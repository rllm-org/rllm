"""Inbound bearer-token auth on SessionRoutingMiddleware.

When the gateway is exposed via a public tunnel, requests must carry
``Authorization: Bearer <token>`` or get rejected before the body even
parses. Without this check, anyone who guesses the tunnel URL can burn
provider credits.
"""

from __future__ import annotations

from typing import Any

import pytest
from rllm_model_gateway.middleware import SessionRoutingMiddleware


class _FakeApp:
    """Records that the inner ASGI app was reached."""

    def __init__(self) -> None:
        self.called = False

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        self.called = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})


def _make_scope(*, auth_header: str | None = None) -> dict[str, Any]:
    headers = []
    if auth_header is not None:
        headers.append((b"authorization", auth_header.encode()))
    return {
        "type": "http",
        "method": "GET",
        "path": "/v1/health",
        "raw_path": b"/v1/health",
        "headers": headers,
    }


async def _call(middleware: SessionRoutingMiddleware, scope: dict[str, Any]) -> tuple[int, bytes]:
    """Drive the middleware once and capture the (status, body) response."""
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    async def send(msg) -> None:  # type: ignore[no-untyped-def]
        sent.append(msg)

    await middleware(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body


@pytest.mark.asyncio
async def test_no_token_configured_means_no_check_runs():
    """Loopback eval (no public URL): inbound_auth_token=None → bypass check entirely."""
    inner = _FakeApp()
    mw = SessionRoutingMiddleware(inner, inbound_auth_token=None)

    status, _ = await _call(mw, _make_scope())  # no Authorization header

    assert status == 200
    assert inner.called is True


@pytest.mark.asyncio
async def test_correct_bearer_token_passes_through():
    inner = _FakeApp()
    mw = SessionRoutingMiddleware(inner, inbound_auth_token="tok_abc")

    status, _ = await _call(mw, _make_scope(auth_header="Bearer tok_abc"))

    assert status == 200
    assert inner.called is True


@pytest.mark.asyncio
async def test_missing_authorization_header_is_401():
    inner = _FakeApp()
    mw = SessionRoutingMiddleware(inner, inbound_auth_token="tok_abc")

    status, body = await _call(mw, _make_scope())

    assert status == 401
    assert b"bearer" in body.lower()
    assert inner.called is False  # short-circuited before the inner app


@pytest.mark.asyncio
async def test_wrong_token_is_401():
    inner = _FakeApp()
    mw = SessionRoutingMiddleware(inner, inbound_auth_token="tok_correct")

    status, _ = await _call(mw, _make_scope(auth_header="Bearer tok_wrong"))

    assert status == 401
    assert inner.called is False


@pytest.mark.asyncio
async def test_non_bearer_scheme_is_401():
    """Basic auth, API-key headers, etc. don't satisfy the bearer requirement."""
    inner = _FakeApp()
    mw = SessionRoutingMiddleware(inner, inbound_auth_token="tok_abc")

    status, _ = await _call(mw, _make_scope(auth_header="Basic dXNlcjpwdw=="))

    assert status == 401
    assert inner.called is False


@pytest.mark.asyncio
async def test_bearer_scheme_is_case_insensitive():
    """Some clients send lower-case 'bearer'. RFC 6750 allows it; reject it and they break."""
    inner = _FakeApp()
    mw = SessionRoutingMiddleware(inner, inbound_auth_token="tok_abc")

    status, _ = await _call(mw, _make_scope(auth_header="bearer tok_abc"))

    assert status == 200


@pytest.mark.asyncio
async def test_websocket_scope_bypasses_check():
    """Auth is for HTTP requests; WS upgrades aren't a thing on this gateway,
    but the middleware should not crash if one shows up."""
    inner = _FakeApp()
    mw = SessionRoutingMiddleware(inner, inbound_auth_token="tok_abc")
    scope = {"type": "websocket", "path": "/", "headers": []}

    async def receive():
        return {"type": "websocket.disconnect"}

    sent = []

    async def send(msg):
        sent.append(msg)

    # Should hit the websocket fast-path that just delegates.
    await mw(scope, receive, send)
    assert inner.called is True

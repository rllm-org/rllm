"""SessionRoutingMiddleware: regex matching + path rewriting."""

from __future__ import annotations

from rllm_model_gateway.middleware import SESSION_PATH_RE


def test_matches_v1_chat_completions():
    m = SESSION_PATH_RE.search("/sessions/abc/v1/chat/completions")
    assert m is not None
    assert m.group(1) == "abc"
    assert m.group(2) == "/v1/chat/completions"


def test_matches_v1_messages():
    m = SESSION_PATH_RE.search("/sessions/sid-1/v1/messages")
    assert m is not None
    assert m.group(1) == "sid-1"
    assert m.group(2) == "/v1/messages"


def test_does_not_match_traces_endpoint():
    """Management endpoints under /sessions/{sid}/... must not match."""
    assert SESSION_PATH_RE.search("/sessions/abc/traces") is None
    assert SESSION_PATH_RE.search("/sessions/abc") is None


def test_does_not_match_root_v1():
    assert SESSION_PATH_RE.search("/v1/chat/completions") is None
    assert SESSION_PATH_RE.search("/sessions") is None


def test_session_id_can_be_uuid():
    m = SESSION_PATH_RE.search("/sessions/12345678-1234-1234-1234-123456789012/v1/responses")
    assert m is not None
    assert m.group(1) == "12345678-1234-1234-1234-123456789012"

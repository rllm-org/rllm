"""Endpoint shapers — wire-format ↔ NormalizedRequest/Response conversion.

Each shaper module exposes:
    NAME: str
    PATH: str                                    # FastAPI route inside the gateway
    UPSTREAM_PATH: str                           # appended to upstream_url in passthrough
    to_normalized_request(body: dict) -> NormalizedRequest
    parse_upstream_response(body: dict) -> NormalizedResponse
    parse_upstream_stream(chunks: list[dict]) -> NormalizedResponse
    from_normalized_response_nonstream(resp, model: str) -> dict
    from_normalized_response_stream(resp, model: str) -> AsyncIterator[str]
"""

from __future__ import annotations

from rllm_model_gateway.endpoints import (
    anthropic_messages,
    chat_completions,
    completions,
    responses,
)

SHAPERS = {
    chat_completions.NAME: chat_completions,
    completions.NAME: completions,
    responses.NAME: responses,
    anthropic_messages.NAME: anthropic_messages,
}

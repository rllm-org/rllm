"""Legacy /v1/completions wire format ↔ normalized."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from rllm_model_gateway.normalized import NormalizedRequest, NormalizedResponse, Usage

NAME = "completions"
PATH = "/v1/completions"
UPSTREAM_PATH = "/completions"

_NON_SAMPLING_KEYS = frozenset({"model", "prompt", "stream", "user", "n", "logprobs", "stream_options"})


def to_normalized_request(body: dict[str, Any]) -> NormalizedRequest:
    prompt = body.get("prompt")
    if isinstance(prompt, list):
        # Legacy completions accepts a list; we only support the first.
        prompt = prompt[0] if prompt else ""
    sampling = {k: v for k, v in body.items() if k not in _NON_SAMPLING_KEYS}
    return NormalizedRequest(prompt=prompt or "", sampling_params=sampling)


def parse_upstream_response(body: dict[str, Any]) -> NormalizedResponse:
    choices = body.get("choices") or [{}]
    choice = choices[0]
    usage_raw = body.get("usage") or {}
    return NormalizedResponse(
        content=choice.get("text", ""),
        finish_reason=choice.get("finish_reason") or "stop",
        usage=Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        ),
    )


def parse_upstream_stream(chunks: list[dict[str, Any]]) -> NormalizedResponse:
    text_parts: list[str] = []
    finish_reason: str | None = None
    usage_raw: dict[str, Any] = {}
    for chunk in chunks:
        choices = chunk.get("choices") or []
        if choices:
            ch = choices[0]
            if isinstance(ch.get("text"), str):
                text_parts.append(ch["text"])
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
        if chunk.get("usage"):
            usage_raw = chunk["usage"]
    return NormalizedResponse(
        content="".join(text_parts),
        finish_reason=finish_reason or "stop",
        usage=Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        ),
    )


def from_normalized_response_nonstream(resp: NormalizedResponse, model: str) -> dict[str, Any]:
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": resp.content,
                "finish_reason": resp.finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.prompt_tokens + resp.usage.completion_tokens,
        },
    }


async def from_normalized_response_stream(resp: NormalizedResponse, model: str) -> AsyncIterator[str]:
    cmpl_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    body = {
        "id": cmpl_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": resp.content,
                "finish_reason": resp.finish_reason,
                "logprobs": None,
            }
        ],
    }
    yield f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    final = {
        "id": cmpl_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.prompt_tokens + resp.usage.completion_tokens,
        },
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

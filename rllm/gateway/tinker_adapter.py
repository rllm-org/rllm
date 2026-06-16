"""Create a local handler from TinkerEngine for the model gateway.

The handler is a plain ``async (dict) -> dict`` callable that translates
OpenAI-format request dicts into ``TinkerEngine.get_model_response()`` calls
and returns responses with embedded token IDs and logprobs in the format
expected by the gateway's ``data_process.py`` extractors.

This replaces the sidecar ``TinkerBackendServer`` with an in-process call,
eliminating the extra HTTP hop and port allocation.
"""

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from rllm.engine.rollout.tinker_engine import TinkerEngine

logger = logging.getLogger(__name__)


def _to_openai_tool_calls(tool_calls: list) -> list[dict[str, Any]]:
    """Convert rLLM ToolCall objects to OpenAI-format tool_calls."""
    result = []
    for i, tc in enumerate(tool_calls):
        name = tc.name if hasattr(tc, "name") else tc.get("name", "")
        args = tc.arguments if hasattr(tc, "arguments") else tc.get("arguments", {})
        if isinstance(args, dict):
            args_str = json.dumps(args)
        else:
            args_str = str(args)
        result.append(
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }
        )
    return result


async def _token_prompt_completion(
    engine: TinkerEngine,
    request_body: dict[str, Any],
    prompt_ids: list[int],
    sampling_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Sample from a pre-tokenized prompt (cumulative-token mode).

    Bypasses message rendering entirely: ``prompt_ids`` is fed straight to the
    engine's token-input sampler. Returns a **completions-style** response dict
    (``choices[0].text`` + ``token_ids``, root ``prompt_token_ids``,
    ``logprobs.token_logprobs``) — the exact shape the gateway's cumulative-turn
    handler extracts token IDs from and translates back to chat format.
    """
    token_output = await engine.get_token_output_from_token_input(prompt_ids, **sampling_kwargs)
    model_output = engine.assemble_model_output(prompt_ids, token_output)

    # Text the agent sees as the assistant turn (same precedence as the chat path).
    text = model_output.content or model_output.text or ""
    out_prompt_ids = list(model_output.prompt_ids) if model_output.prompt_ids else list(prompt_ids)
    completion_ids = list(model_output.completion_ids) if model_output.completion_ids else []
    logprobs = model_output.logprobs or []
    finish_reason = model_output.finish_reason or "stop"
    prompt_len = model_output.prompt_length or len(out_prompt_ids)
    completion_len = model_output.completion_length or len(completion_ids)

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request_body.get("model", getattr(engine, "model_name", "default")),
        "choices": [
            {
                "index": 0,
                "text": text,
                "token_ids": completion_ids,
                "finish_reason": finish_reason,
                "logprobs": {"token_logprobs": logprobs},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_len,
            "total_tokens": prompt_len + completion_len,
        },
        "prompt_token_ids": out_prompt_ids,
        "weight_version": getattr(model_output, "weight_version", None),
    }


def create_tinker_handler(engine: TinkerEngine) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Return an async handler that calls TinkerEngine in-process.

    The returned callable accepts an OpenAI chat completion request dict and
    returns an OpenAI chat completion response dict with token extensions
    (``prompt_token_ids``, ``token_ids``, ``logprobs``) consistent with vLLM.
    """

    async def handler(request_body: dict[str, Any]) -> dict[str, Any]:
        # Sampling params shared by the chat and pre-tokenized paths.
        sampling_kwargs: dict[str, Any] = {}
        if request_body.get("temperature") is not None:
            sampling_kwargs["temperature"] = request_body["temperature"]
        if request_body.get("top_p") is not None:
            sampling_kwargs["top_p"] = request_body["top_p"]
        if request_body.get("top_k") is not None:
            sampling_kwargs["top_k"] = request_body["top_k"]
        if request_body.get("max_tokens") is not None:
            sampling_kwargs["max_tokens"] = request_body["max_tokens"]
        if request_body.get("max_completion_tokens") is not None:
            sampling_kwargs["max_completion_tokens"] = request_body["max_completion_tokens"]

        # Cumulative-token-mode path: the gateway rewrites turn 2+ to a
        # completions-style request whose ``prompt`` is raw token IDs built by
        # ``renderers.bridge_to_next_turn`` (= prior turns' prompt+completion
        # tokens + the new messages). Sample straight from those tokens — no
        # re-render, no re-tokenization — so the sequence the optimizer trains on
        # is byte-for-byte what was generated. Mirrors the vLLM /v1/completions
        # path the HTTP backend uses for the same feature.
        prompt = request_body.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            return await _token_prompt_completion(engine, request_body, prompt, sampling_kwargs)

        messages = request_body.get("messages", [])
        tools = request_body.get("tools", [])
        kwargs = dict(sampling_kwargs)
        if tools:
            kwargs["tools"] = tools

        model_output = await engine.get_model_response(messages, **kwargs)

        response_text = model_output.content or model_output.text or ""
        prompt_ids = list(model_output.prompt_ids) if model_output.prompt_ids else []
        completion_ids = list(model_output.completion_ids) if model_output.completion_ids else []
        logprobs = model_output.logprobs or []
        finish_reason = model_output.finish_reason or "stop"

        response_message: dict[str, Any] = {"role": "assistant", "content": response_text}
        if model_output.reasoning:
            response_message["reasoning"] = model_output.reasoning
        if model_output.tool_calls:
            response_message["tool_calls"] = _to_openai_tool_calls(model_output.tool_calls)
            if finish_reason == "stop":
                finish_reason = "tool_calls"

        prompt_len = model_output.prompt_length or len(prompt_ids)
        completion_len = model_output.completion_length or len(completion_ids)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_body.get("model", getattr(engine, "model_name", "default")),
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": finish_reason,
                    "token_ids": completion_ids,
                    "routing_matrices": getattr(model_output, "routing_matrices", None),
                    "logprobs": {
                        "content": [{"logprob": lp} for lp in logprobs],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_len,
                "total_tokens": prompt_len + completion_len,
            },
            "prompt_token_ids": prompt_ids,
            "weight_version": getattr(model_output, "weight_version", None),
        }

    return handler

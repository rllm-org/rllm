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

from rllm.experimental.rollout.tinker_engine import TinkerEngine

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


def create_tinker_handler(engine: TinkerEngine) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Return an async handler that calls TinkerEngine in-process.

    The returned callable accepts an OpenAI chat completion request dict and
    returns an OpenAI chat completion response dict with token extensions
    (``prompt_token_ids``, ``token_ids``, ``logprobs``) consistent with vLLM.
    """

    async def handler(request_body: dict[str, Any]) -> dict[str, Any]:
        messages = request_body.get("messages", [])
        tools = request_body.get("tools", [])

        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
        if request_body.get("temperature") is not None:
            kwargs["temperature"] = request_body["temperature"]
        if request_body.get("top_p") is not None:
            kwargs["top_p"] = request_body["top_p"]
        if request_body.get("max_tokens") is not None:
            kwargs["max_tokens"] = request_body["max_tokens"]
        if request_body.get("max_completion_tokens") is not None:
            kwargs["max_completion_tokens"] = request_body["max_completion_tokens"]

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
        }

    return handler

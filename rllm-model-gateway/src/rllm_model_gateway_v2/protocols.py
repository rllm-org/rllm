import json
import uuid
from collections.abc import Iterator
from typing import Any

from rllm_model_gateway_v2.contracts import APIProtocol, CanonicalOutput, CanonicalRequest
from rllm_model_gateway_v2.errors import GatewayError


_HEAD_FIELDS = {"model", "stream", "stream_options", "prompt", "messages", "tools"}


def normalize_request(
    protocol: APIProtocol,
    session_id: str,
    body: dict[str, Any],
) -> CanonicalRequest:
    request_id = f"req_{uuid.uuid4().hex}"
    sampling = {key: value for key, value in body.items() if key not in _HEAD_FIELDS}
    tools = body.get("tools") or []
    if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
        raise GatewayError("tools must be an array of objects")

    common: dict[str, Any] = {
        "request_id": request_id,
        "session_id": session_id,
        "sampling_params": sampling,
        "tools": tools,
    }

    if protocol == APIProtocol.COMPLETIONS:
        prompt = body.get("prompt")
        if isinstance(prompt, str):
            return CanonicalRequest(prompt=prompt, **common)
        if isinstance(prompt, list) and all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in prompt):
            return CanonicalRequest(prompt_token_ids=prompt, **common)
        raise GatewayError("completions require one string or token-id prompt")

    if protocol == APIProtocol.CHAT_COMPLETIONS:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise GatewayError("messages must be a non-empty array")
        if any(not isinstance(message, dict) or not isinstance(message.get("role"), str) for message in messages):
            raise GatewayError("each message must be an object with a role")
        return CanonicalRequest(messages=messages, **common)

    raise GatewayError(f"unsupported protocol: {protocol}")


def response_payload(
    protocol: APIProtocol,
    output: CanonicalOutput,
    model: str,
    created_at: int,
) -> dict[str, Any]:
    if protocol == APIProtocol.COMPLETIONS:
        return _completion_payload(output, model, created_at)
    return _chat_payload(output, model, created_at)


def _usage(output: CanonicalOutput) -> dict[str, int]:
    prompt = output.prompt_tokens
    completion = output.completion_tokens
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _completion_payload(output: CanonicalOutput, model: str, created_at: int) -> dict[str, Any]:
    return {
        "id": output.request_id,
        "object": "text_completion",
        "created": created_at,
        "model": model,
        "choices": [{"index": 0, "text": output.text, "logprobs": None, "finish_reason": output.finish_reason}],
        "usage": _usage(output),
    }


def _chat_payload(output: CanonicalOutput, model: str, created_at: int) -> dict[str, Any]:
    return {
        "id": output.request_id,
        "object": "chat.completion",
        "created": created_at,
        "model": model,
        "choices": [{"index": 0, "message": _message(output), "logprobs": None, "finish_reason": output.finish_reason}],
        "usage": _usage(output),
    }


def stream_events(
    protocol: APIProtocol,
    output: CanonicalOutput,
    model: str,
    created_at: int,
    include_usage: bool = False,
) -> Iterator[str]:
    if protocol == APIProtocol.COMPLETIONS:
        yield from _completion_stream(output, model, created_at, include_usage)
    else:
        yield from _chat_stream(output, model, created_at, include_usage)
    if include_usage:
        yield _sse(
            {
                "id": output.request_id,
                "object": "text_completion" if protocol == APIProtocol.COMPLETIONS else "chat.completion.chunk",
                "created": created_at,
                "model": model,
                "choices": [],
                "usage": _usage(output),
            }
        )
    yield "data: [DONE]\n\n"


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def _stream_payload(payload: dict[str, Any], include_usage: bool) -> dict[str, Any]:
    if include_usage:
        payload["usage"] = None
    return payload


def _completion_stream(output: CanonicalOutput, model: str, created_at: int, include_usage: bool) -> Iterator[str]:
    base = {"id": output.request_id, "object": "text_completion", "created": created_at, "model": model}
    yield _sse(_stream_payload({**base, "choices": [{"index": 0, "text": output.text, "logprobs": None, "finish_reason": None}]}, include_usage))
    yield _sse(_stream_payload({**base, "choices": [{"index": 0, "text": "", "logprobs": None, "finish_reason": output.finish_reason}]}, include_usage))


def _chat_stream(output: CanonicalOutput, model: str, created_at: int, include_usage: bool) -> Iterator[str]:
    base = {"id": output.request_id, "object": "chat.completion.chunk", "created": created_at, "model": model}
    delta: dict[str, Any] = {"role": "assistant"}
    if output.reasoning_content is not None:
        delta["reasoning_content"] = output.reasoning_content
    if output.content is not None:
        delta["content"] = output.content
    if output.tool_calls:
        delta["tool_calls"] = [
            {**tool_call, "index": tool_index}
            for tool_index, tool_call in enumerate(output.tool_calls)
        ]
    yield _sse(_stream_payload({**base, "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": None}]}, include_usage))
    yield _sse(_stream_payload({**base, "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": output.finish_reason}]}, include_usage))


def _message(output: CanonicalOutput) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": output.content}
    if output.reasoning_content is not None:
        message["reasoning_content"] = output.reasoning_content
    if output.tool_calls:
        message["tool_calls"] = output.tool_calls
    return message


def error_payload(message: str, error_type: str = "server_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "param": None, "code": None}}

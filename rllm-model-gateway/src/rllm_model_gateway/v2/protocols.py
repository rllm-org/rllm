import json
import uuid
from collections.abc import Iterator
from typing import Any

from rllm_model_gateway.v2.types import APIProtocol, GatewayError, GatewayRequest, GatewayResponse

PROTOCOL_ONLY_FIELDS = {
    "model",
    "stream",
    "stream_options",
    "prompt",
    "messages",
    "tools",
    "user",
    "store",
    "metadata",
    "tool_choice",
    "logprobs",
    "top_logprobs",
}


def normalize_request(
    protocol: APIProtocol,
    session_id: str,
    body: dict[str, Any],
) -> GatewayRequest:
    request_id = f"req_{uuid.uuid4().hex}"
    sampling = {key: value for key, value in body.items() if key not in PROTOCOL_ONLY_FIELDS}

    common: dict[str, Any] = {
        "request_id": request_id,
        "session_id": session_id,
        "sampling_params": sampling,
    }

    if protocol == APIProtocol.COMPLETIONS:
        if "messages" in body:
            raise GatewayError("completions do not accept messages")
        if "tools" in body:
            raise GatewayError("completions do not accept tools")
        prompt = body.get("prompt")
        if isinstance(prompt, str):
            return GatewayRequest(prompt=prompt, tools=[], **common)
        if isinstance(prompt, list) and all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in prompt):
            return GatewayRequest(prompt_token_ids=prompt, tools=[], **common)
        raise GatewayError("completions require one string or token-id prompt")

    if protocol == APIProtocol.CHAT_COMPLETIONS:
        if "prompt" in body:
            raise GatewayError("chat completions do not accept prompt")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise GatewayError("messages must be a non-empty array")
        if any(not isinstance(message, dict) or not isinstance(message.get("role"), str) for message in messages):
            raise GatewayError("each message must be an object with a role")
        return GatewayRequest(messages=messages, tools=_validate_tools(body.get("tools")), **common)

    raise GatewayError(f"unsupported protocol: {protocol}")


def _validate_tools(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise GatewayError("tools must be an array")
    tools: list[dict[str, Any]] = []
    for index, tool in enumerate(value):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise GatewayError(f"tools[{index}] must be a function tool")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise GatewayError(f"tools[{index}].function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise GatewayError(f"tools[{index}].function.name must be a non-empty string")
        description = function.get("description")
        if description is not None and not isinstance(description, str):
            raise GatewayError(f"tools[{index}].function.description must be a string")
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise GatewayError(f"tools[{index}].function.parameters must be an object")
        tools.append(tool)
    return tools


def response_payload(
    protocol: APIProtocol,
    output: GatewayResponse,
    model: str,
    created_at: int,
) -> dict[str, Any]:
    if protocol == APIProtocol.COMPLETIONS:
        return _completion_payload(output, model, created_at)
    return _chat_payload(output, model, created_at)


def _usage(output: GatewayResponse) -> dict[str, int]:
    prompt = output.prompt_tokens
    completion = output.completion_tokens
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _completion_payload(output: GatewayResponse, model: str, created_at: int) -> dict[str, Any]:
    return {
        "id": output.request_id,
        "object": "text_completion",
        "created": created_at,
        "model": model,
        "choices": [{"index": 0, "text": output.text, "logprobs": None, "finish_reason": output.finish_reason}],
        "usage": _usage(output),
    }


def _chat_payload(output: GatewayResponse, model: str, created_at: int) -> dict[str, Any]:
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
    output: GatewayResponse,
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


def _completion_stream(output: GatewayResponse, model: str, created_at: int, include_usage: bool) -> Iterator[str]:
    base = {"id": output.request_id, "object": "text_completion", "created": created_at, "model": model}
    yield _sse(_stream_payload({**base, "choices": [{"index": 0, "text": output.text, "logprobs": None, "finish_reason": None}]}, include_usage))
    yield _sse(_stream_payload({**base, "choices": [{"index": 0, "text": "", "logprobs": None, "finish_reason": output.finish_reason}]}, include_usage))


def _chat_stream(output: GatewayResponse, model: str, created_at: int, include_usage: bool) -> Iterator[str]:
    base = {"id": output.request_id, "object": "chat.completion.chunk", "created": created_at, "model": model}
    delta: dict[str, Any] = {"role": "assistant"}
    if output.reasoning_content is not None:
        delta["reasoning_content"] = output.reasoning_content
    if output.content is not None:
        delta["content"] = output.content
    if output.tool_calls:
        delta["tool_calls"] = [{**tool_call, "index": tool_index} for tool_index, tool_call in enumerate(output.tool_calls)]
    yield _sse(_stream_payload({**base, "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": None}]}, include_usage))
    yield _sse(_stream_payload({**base, "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": output.finish_reason}]}, include_usage))


def _message(output: GatewayResponse) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": output.content}
    if output.reasoning_content is not None:
        message["reasoning_content"] = output.reasoning_content
    if output.tool_calls:
        message["tool_calls"] = output.tool_calls
    return message


def error_payload(message: str, error_type: str = "server_error", code: int | str | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "param": None, "code": code}}

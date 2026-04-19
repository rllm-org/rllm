"""Wrap a ``RolloutEngine`` as a gateway adapter.

The gateway calls ``async fn(NormalizedRequest) -> NormalizedResponse``. Every
``RolloutEngine`` subclass exposes ``async get_model_response(messages, **kwargs)
-> ModelOutput`` with the same shape, so a single generic wrapper handles
Tinker, Verl, and any future engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rllm_model_gateway import (
    AdapterError,
    AdapterFn,
    Message,
    NormalizedRequest,
    NormalizedResponse,
    Usage,
)
from rllm_model_gateway import (
    ToolCall as GatewayToolCall,
)

if TYPE_CHECKING:
    from rllm.experimental.rollout import RolloutEngine
    from rllm.experimental.rollout.rollout_engine import ModelOutput


def create_engine_adapter(engine: RolloutEngine) -> AdapterFn:
    """Build a gateway adapter that delegates to the ``RolloutEngine``.

    Chat-shaped requests (messages) go through ``get_model_response`` which
    applies the chat template. Raw-text requests (``/v1/completions``) take
    the token-in/token-out path directly — tokenize the prompt, generate,
    assemble ``ModelOutput`` — bypassing the chat template so the returned
    token IDs align with the raw prompt bytes.
    """

    async def adapter(req: NormalizedRequest) -> NormalizedResponse:
        kwargs: dict[str, Any] = dict(req.kwargs)
        if req.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in req.tools
            ]

        if req.prompt is not None:
            model_output = await _run_completions(engine, req.prompt, kwargs)
        else:
            messages = [_message_to_dict(m) for m in (req.messages or [])]
            model_output = await engine.get_model_response(messages, **kwargs)
        return _model_output_to_normalized_response(model_output)

    return adapter


async def _run_completions(engine: RolloutEngine, prompt: str, kwargs: dict[str, Any]) -> ModelOutput:
    """Raw-text → raw-tokens path for /v1/completions.

    Tokenizes with the engine's own tokenizer, calls the TITO interface so
    no chat template is applied, then assembles a ``ModelOutput`` via the
    engine's own converter.
    """
    if engine.tokenizer is None:
        raise AdapterError("engine has no tokenizer; /v1/completions requires one", 400)
    weight_version = engine.weight_version
    token_ids: list[int] = engine.tokenizer.encode(prompt)
    token_output = await engine.get_token_output_from_token_input(token_ids, **kwargs)
    model_output = engine.assemble_model_output(token_ids, token_output)
    model_output.weight_version = weight_version
    return model_output


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def _message_to_dict(m: Message) -> dict[str, Any]:
    """Convert a gateway Message to the dict shape RolloutEngine expects."""
    out: dict[str, Any] = {"role": m.role}
    if m.content is not None:
        out["content"] = m.content
    if m.reasoning is not None:
        out["reasoning"] = m.reasoning
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in m.tool_calls
        ]
    if m.tool_call_id is not None:
        out["tool_call_id"] = m.tool_call_id
    if m.name is not None:
        out["name"] = m.name
    return out


def _model_output_to_normalized_response(mo: ModelOutput) -> NormalizedResponse:
    extras: dict[str, Any] = {}
    if mo.prompt_ids is not None:
        extras["prompt_ids"] = list(mo.prompt_ids)
    if mo.completion_ids is not None:
        extras["completion_ids"] = list(mo.completion_ids)
    if mo.logprobs is not None:
        extras["logprobs"] = list(mo.logprobs)
    if mo.prompt_logprobs is not None:
        extras["prompt_logprobs"] = list(mo.prompt_logprobs)
    if mo.routing_matrices is not None:
        extras["routing_matrices"] = list(mo.routing_matrices)

    tool_calls: list[GatewayToolCall] = []
    for tc in mo.tool_calls or []:
        if tc.arguments_raw is not None:
            args = tc.arguments_raw
        elif isinstance(tc.arguments, dict):
            args = _json_dumps(tc.arguments)
        else:
            args = ""
        tool_calls.append(GatewayToolCall(name=tc.name, arguments=args))

    metadata: dict[str, Any] = {}
    if mo.weight_version is not None:
        metadata["weight_version"] = mo.weight_version

    return NormalizedResponse(
        text=mo.text,
        content=mo.content or "",
        reasoning=mo.reasoning,
        tool_calls=tool_calls,
        finish_reason=mo.finish_reason or "stop",
        usage=Usage(
            prompt_tokens=mo.prompt_length or (len(mo.prompt_ids) if mo.prompt_ids else 0),
            completion_tokens=mo.completion_length or (len(mo.completion_ids) if mo.completion_ids else 0),
        ),
        extras=extras,
        metrics=dict(mo.metrics) if mo.metrics else {},
        metadata=metadata,
    )


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)

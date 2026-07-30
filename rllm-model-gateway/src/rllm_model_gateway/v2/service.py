import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from rllm_model_gateway.v2.backend import GenerationBackend
from rllm_model_gateway.v2.types import (
    GatewayRequest,
    GatewayResponse,
    GatewayError,
    SessionTraces,
    TokenInput,
    TokenOutput,
    TraceRecord,
)
from rllm_model_gateway.v2.tokenization import TokenizationService


@dataclass
class SessionState:
    traces: SessionTraces
    sampling_params: dict[str, Any] = field(default_factory=dict)
    message_count: int = 0
    render_context_hash: str | None = None


class GatewayService:
    def __init__(self, tokenization: TokenizationService, backend: GenerationBackend, cumulative: bool = False) -> None:
        self._tokenization = tokenization
        self._backend = backend
        self._cumulative = cumulative
        self._sessions: dict[str, SessionState] = {}

    def create_session(
        self,
        session_id: str,
        sampling_params: dict[str, Any] | None = None,
    ) -> None:
        if session_id in self._sessions:
            raise GatewayError(f"session {session_id!r} already exists", 409)
        state = SessionState(
            traces=SessionTraces(session_id=session_id),
            sampling_params=dict(sampling_params or {}),
        )
        self._sessions[session_id] = state

    async def generate(self, request: GatewayRequest) -> GatewayResponse:
        started_at = time.time()
        state = self._require_session(request.session_id)
        sampling_params = dict(request.sampling_params)
        sampling_params.update(state.sampling_params)
        output_count = sampling_params.pop("n", 1)
        if isinstance(output_count, bool) or not isinstance(output_count, int) or output_count != 1:
            raise GatewayError("requests require n=1")
        stop_token_ids = self._tokenization.stop_token_ids()
        if stop_token_ids:
            sampling_params["stop_token_ids"] = stop_token_ids
        prompt_token_ids = self._get_prompt_token_ids(state, request)
        token_input = TokenInput(
            session_id=request.session_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
        token_output: TokenOutput = await self._backend.generate(token_input)
        parsed = self._tokenization.parse_completion(token_output.completion_token_ids, request.tools)
        tool_calls = parsed["tool_calls"]
        finish_reason = "tool_calls" if tool_calls else token_output.finish_reason
        response = GatewayResponse(
            request_id=request.request_id,
            text=self._tokenization.decode(token_output.completion_token_ids),
            content=parsed["content"],
            reasoning_content=parsed["reasoning_content"],
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=len(token_output.completion_token_ids),
        )
        trace = TraceRecord(
            request=request,
            response=response,
            input=token_input,
            output=token_output,
            started_at=started_at,
            completed_at=time.time(),
        )
        state.traces.traces.append(trace)
        if self._cumulative and request.messages:
            state.message_count = len(request.messages)
            state.render_context_hash = _fingerprint({"messages": request.messages, "tools": request.tools})
        else:
            state.message_count = 0
            state.render_context_hash = None
        return response

    def get_session_traces(self, session_id: str) -> dict[str, Any]:
        state = self._require_session(session_id)
        return state.traces.model_dump(mode="json")

    def delete_session(self, session_id: str) -> None:
        if self._sessions.pop(session_id, None) is None:
            raise GatewayError(f"session {session_id!r} was not found", 404)

    async def close(self) -> None:
        await self._backend.close()

    def _get_prompt_token_ids(self, state: SessionState, request: GatewayRequest) -> list[int]:
        if request.prompt_token_ids is not None:
            return list(request.prompt_token_ids)
        if request.prompt is not None:
            return self._tokenization.encode(request.prompt)

        if self._cumulative and state.traces.traces:
            previous = state.traces.traces[-1]
            if (
                0 < state.message_count < len(request.messages)
                and _fingerprint({"messages": request.messages[: state.message_count], "tools": request.tools})
                == state.render_context_hash
            ):
                new_messages = [message for message in request.messages[state.message_count :] if message.get("role") != "assistant"]
                if new_messages:
                    bridged = self._tokenization.bridge(
                        previous.input.prompt_token_ids,
                        previous.output.completion_token_ids,
                        new_messages,
                        request.tools,
                    )
                    if bridged is not None:
                        return bridged

        return self._tokenization.render(request.messages, request.tools)

    def _require_session(self, session_id: str) -> SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            raise GatewayError(f"session {session_id!r} was not found", 404)
        return state


def _fingerprint(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

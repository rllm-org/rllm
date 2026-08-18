import logging
import time
from dataclasses import dataclass, field
from typing import Any

from rllm_model_gateway.v2.inference import InferenceClient
from rllm_model_gateway.v2.tokenization import TokenizationService
from rllm_model_gateway.v2.types import (
    GatewayError,
    GatewayRequest,
    GatewayResponse,
    SessionTraces,
    TokenInput,
    TokenOutput,
    TraceLineage,
    TraceRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    traces: SessionTraces
    sampling_params: dict[str, Any] = field(default_factory=dict)


class GatewayService:
    def __init__(self, tokenization: TokenizationService, inference_client: InferenceClient) -> None:
        self._tokenization = tokenization
        self._inference_client = inference_client
        self._sessions: dict[str, SessionState] = {}

    def create_session(
        self,
        session_id: str,
        sampling_params: dict[str, Any] | None = None,
    ) -> None:
        if session_id in self._sessions:
            raise GatewayError(f"session {session_id!r} already exists", 409, "conflict_error")
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
        try:
            prompt_token_ids, lineage = self._get_prompt(state, request)
        except GatewayError:
            raise
        except (TypeError, ValueError) as exc:
            raise GatewayError(f"invalid request: {exc}") from exc
        token_input = TokenInput(
            routing_key=request.session_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
        token_output: TokenOutput = await self._inference_client.generate(token_input)
        text = self._tokenization.decode(token_output.completion_token_ids)
        if request.messages:
            parsed = self._tokenization.parse_completion(token_output.completion_token_ids, request.tools)
            content = parsed["content"]
            reasoning_content = parsed["reasoning_content"]
            tool_calls = parsed["tool_calls"]
        else:
            content = None
            reasoning_content = None
            tool_calls = []
        finish_reason = "tool_calls" if tool_calls else token_output.finish_reason
        response = GatewayResponse(
            request_id=request.request_id,
            text=text,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=len(token_output.completion_token_ids),
        )
        trace = TraceRecord(
            lineage=lineage,
            request=request,
            response=response,
            input=token_input,
            output=token_output,
            started_at=started_at,
            completed_at=time.time(),
        )
        state.traces.traces.append(trace)
        return response

    def get_session_traces(self, session_id: str) -> dict[str, Any]:
        state = self._require_session(session_id)
        return state.traces.model_dump(mode="json")

    def delete_session(self, session_id: str) -> None:
        if self._sessions.pop(session_id, None) is None:
            raise GatewayError(f"session {session_id!r} was not found", 404, "not_found_error")

    async def close(self) -> None:
        await self._inference_client.close()

    def _get_prompt(self, state: SessionState, request: GatewayRequest) -> tuple[list[int], TraceLineage]:
        if request.prompt_token_ids is not None:
            return list(request.prompt_token_ids), _build_lineage(request.request_id, None)
        if request.prompt is not None:
            return self._tokenization.encode(request.prompt), _build_lineage(request.request_id, None)

        matched_trace = _match_trace(state.traces.traces, request)
        if matched_trace is not None:
            new_messages = request.messages[len(matched_trace.request.messages) + 1 :]
            if new_messages:
                bridged = self._tokenization.bridge(
                    matched_trace.input.prompt_token_ids,
                    matched_trace.output.completion_token_ids,
                    new_messages,
                    request.tools,
                )
                if bridged is not None:
                    return bridged, _build_lineage(request.request_id, matched_trace)

        prompt_token_ids = self._tokenization.render(request.messages, request.tools)
        return prompt_token_ids, _build_lineage(request.request_id, None)

    def _require_session(self, session_id: str) -> SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            raise GatewayError(f"session {session_id!r} was not found", 404, "not_found_error")
        return state


def _build_lineage(request_id: str, parent: TraceRecord | None) -> TraceLineage:
    if parent is None:
        return TraceLineage(
            parent_request_id=None,
            root_request_id=request_id,
        )
    return TraceLineage(
        parent_request_id=parent.request.request_id,
        root_request_id=parent.lineage.root_request_id,
    )


def _match_trace(traces: list[TraceRecord], request: GatewayRequest) -> TraceRecord | None:
    matches: list[tuple[int, TraceRecord]] = []
    for trace in traces:
        if not trace.request.messages or trace.request.tools != request.tools:
            continue
        previous_length = len(trace.request.messages)
        if previous_length >= len(request.messages) or request.messages[:previous_length] != trace.request.messages:
            continue
        assistant_message = request.messages[previous_length]
        if not _matches_response(assistant_message, trace.response):
            continue
        matches.append((previous_length, trace))

    if not matches:
        return None
    longest_message_count = max(message_count for message_count, _ in matches)
    longest_matches = [trace for message_count, trace in matches if message_count == longest_message_count]
    if len(longest_matches) > 1:
        logger.warning(
            "Request %s in session %s has %d lineage matches with the same message length (%d); creating a new root",
            request.request_id,
            request.session_id,
            len(longest_matches),
            longest_message_count,
        )
        return None
    return longest_matches[0]


def _matches_response(message: dict[str, Any], response: GatewayResponse) -> bool:
    if message.get("role") != "assistant":
        return False
    if not any(field in message for field in ("content", "reasoning_content", "tool_calls")):
        return False
    if "content" in message and message["content"] != response.content:
        return False
    reasoning = message.get("reasoning_content")
    if reasoning is not None and reasoning != response.reasoning_content:
        return False
    if "tool_calls" in message and message["tool_calls"] != response.tool_calls:
        return False
    return True

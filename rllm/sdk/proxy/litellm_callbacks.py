"""LiteLLM callbacks for parameter injection and tracing."""

from __future__ import annotations

import logging
from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponse, ModelResponseStream

from rllm.sdk.tracers import SqliteTracer

logger = logging.getLogger(__name__)


class SamplingParametersCallback(CustomLogger):
    """Inject sampling parameters before LiteLLM sends requests.

    Adds logprobs and top_logprobs to all requests.
    Only adds return_token_ids for vLLM-compatible backends (not OpenAI/Anthropic).
    """

    def __init__(self, add_return_token_ids: bool = False, add_logprobs: bool = False):
        super().__init__()
        self.add_return_token_ids = add_return_token_ids
        self.add_logprobs = add_logprobs

    async def async_pre_call_hook(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = kwargs.get("data") or (args[2] if len(args) > 2 else {})

        if self.add_logprobs:
            data["logprobs"] = True

        if self.add_return_token_ids:
            data["return_token_ids"] = True

        return data


class TracingCallback(CustomLogger):
    """Log LLM calls to tracer right after provider response.

    Uses LiteLLM's async_post_call_success_hook which fires at the proxy level,
    once per HTTP request, immediately before the response is sent to the user.
    This guarantees we log with the actual response object, while still being
    pre-send, and avoids duplicate logging from nested deployment calls.
    """

    def __init__(self, tracer: SqliteTracer, *, await_persistence: bool = False):
        super().__init__()
        self.tracer = tracer
        self._await_persistence = await_persistence

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: ModelResponse | ModelResponseStream,
    ) -> Any:
        """Called once per HTTP request at proxy level, before response is sent to user.

        This hook is called only once per HTTP request
        It has access to the actual response object
        and runs synchronously before the HTTP response is returned.

        Uses litellm_call_id for deduplication to ensure we only log once per request.
        """

        metadata = data.get("metadata", {}).get("requester_metadata", {}).get("rllm_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        model = data.get("model", "unknown") if isinstance(data, dict) else "unknown"
        messages = data.get("messages", []) if isinstance(data, dict) else []

        # Latency best-effort: prefer provider response_ms if available
        latency_ms: float = 0.0
        latency_ms = float(getattr(response, "response_ms", 0.0) or 0.0)

        usage = getattr(response, "usage", None)
        tokens = {
            "prompt": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total": getattr(usage, "total_tokens", 0) if usage else 0,
        }

        if hasattr(response, "model_dump"):
            response_payload: Any = response.model_dump()
        elif isinstance(response, dict):
            response_payload = response
        else:
            response_payload = {"text": str(response)}

        # Extract the response ID from the LLM provider to use as trace_id/context_id
        # This ensures the context_id matches the actual completion ID from the provider
        response_id = response_payload.get("id", None)

        # Extract session_uids and session_name from metadata (sent from client)
        session_uids = metadata.get("session_uids", None)
        session_name = metadata.get("session_name")

        log_kwargs = dict(
            name=f"proxy/{model}",
            model=model,
            input={"messages": messages},
            output=response_payload,
            metadata=metadata,
            session_name=session_name,
            latency_ms=latency_ms,
            tokens=tokens,
            trace_id=response_id,
            session_uids=session_uids,
        )

        if self._await_persistence:
            await self.tracer.log_llm_call_sync(**log_kwargs)
        else:
            self.tracer.log_llm_call(**log_kwargs)

        # Return response unchanged
        return response

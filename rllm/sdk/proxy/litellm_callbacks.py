"""LiteLLM callbacks for parameter injection and tracing."""

from __future__ import annotations

from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponse, ModelResponseStream

from rllm.sdk.tracers import SqliteTracer


class SamplingParametersCallback(CustomLogger):
    """Inject sampling parameters and metadata before LiteLLM sends requests.

    Adds logprobs and top_logprobs to all requests.
    Only adds return_token_ids for vLLM-compatible backends (not OpenAI/Anthropic).
    Injects metadata from request state if available.
    """

    def __init__(self, add_return_token_ids: bool = False):
        super().__init__()
        self.add_return_token_ids = add_return_token_ids

    async def async_pre_call_hook(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = kwargs.get("data") or (args[2] if len(args) > 2 else {})
        model = data.get("model", "")

        # Request token-level logprobs; do not force top_logprobs list
        # result = {**data, "logprobs": True}
        result = {**data}

        # Extract litellm_params to check backend type
        litellm_params = kwargs.get("litellm_params", {})

        # Only add return_token_ids if explicitly enabled AND model supports it
        if self.add_return_token_ids and self._supports_token_ids(model, litellm_params):
            result["return_token_ids"] = True

        # Inject metadata from request state if available
        proxy_server_request = litellm_params.get("proxy_server_request")
        if proxy_server_request:
            request_state = getattr(proxy_server_request, "state", None)
            if request_state:
                rllm_metadata = getattr(request_state, "rllm_metadata", None)
                if rllm_metadata:
                    result["metadata"] = {**result.get("metadata", {}), **rllm_metadata}

        return result

    @staticmethod
    def _supports_token_ids(model: str, litellm_params: dict[str, Any] | None = None) -> bool:
        """Check if model supports return_token_ids parameter.

        Only vLLM backends support this. We detect vLLM by checking the
        backend model type in litellm_params (configured by proxy_manager).

        Args:
            model: The model name from the request (unused, kept for compatibility)
            litellm_params: LiteLLM parameters containing backend info

        Returns:
            True if backend is vLLM, False otherwise
        """
        # Check if backend is vLLM (configured as hosted_vllm/* by proxy_manager)
        if litellm_params:
            backend_model = litellm_params.get("model", "")
            if "vllm" in backend_model.lower() or "hosted_vllm" in backend_model.lower():
                return True
        return False


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
        raw_meta_from_data = data.get("metadata", {}) if isinstance(data, dict) else {}
        metadata = raw_meta_from_data.get("requester_metadata") if isinstance(raw_meta_from_data, dict) else None
        if not isinstance(metadata, dict):
            metadata = raw_meta_from_data if isinstance(raw_meta_from_data, dict) else {}

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

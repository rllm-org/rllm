"""Extremely small wrapper that logs chat completions via the tracer."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

from rllm.sdk.session import get_current_metadata, get_current_session_name


class _SimpleTrackedChatClientBase:
    tracer: Any
    default_model: str | None

    @staticmethod
    def _merge_args(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        if args:
            if len(args) == 1 and isinstance(args[0], Mapping):
                merged = dict(args[0])
                merged.update(kwargs)
                return merged
            raise TypeError("Positional arguments are not supported; pass keyword arguments.")
        return dict(kwargs)

    @staticmethod
    def _extract_completion_tokens(response_payload: Mapping[str, Any]) -> list[int] | None:
        choices = response_payload.get("choices") or []
        if not choices:
            return None
        choice0 = choices[0]
        output_ids = choice0.get("output_token_ids")
        if isinstance(output_ids, list):
            return [int(tok) for tok in output_ids]
        logprobs = choice0.get("logprobs")
        if isinstance(logprobs, Mapping):
            token_ids = logprobs.get("token_ids")
            if isinstance(token_ids, list):
                return [int(tok) for tok in token_ids]
        return None

    def _log_trace(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_payload: Mapping[str, Any],
        completion_token_ids: list[int] | None,
        metadata_overrides: Mapping[str, Any],
        latency_ms: float,
    ) -> None:
        if not self.tracer:
            return

        session_name = get_current_session_name()
        context_metadata = dict(get_current_metadata() or {})

        metadata: dict[str, Any] = dict(context_metadata)
        metadata.update(dict(metadata_overrides or {}))
        metadata["token_ids"] = {"prompt": []}
        if completion_token_ids:
            metadata["token_ids"]["completion"] = completion_token_ids

        usage = response_payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

        tokens_summary = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        }

        self.tracer.log_llm_call(
            name="simple.chat.completions.create",
            model=model,
            input={"messages": messages},
            output=response_payload,
            session_name=session_name,
            metadata=metadata,
            latency_ms=latency_ms,
            tokens=tokens_summary,
        )


@dataclass
class _ChatCompletions:
    parent: SimpleTrackedChatClient

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = self.parent._merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        start = time.perf_counter()
        response = self.parent._client.chat.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        response_dict = response.model_dump()

        completion_token_ids = self.parent._extract_completion_tokens(response_dict)

        self.parent._log_trace(
            model=model,
            messages=messages,
            response_payload=response_dict,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=latency_ms,
        )

        return response


@dataclass
class _ChatNamespace:
    parent: SimpleTrackedChatClient

    @property
    def completions(self) -> _ChatCompletions:
        return _ChatCompletions(self.parent)


@dataclass
class _Completions:
    parent: SimpleTrackedChatClient

    def create(self, *args: Any, **kwargs: Any) -> Completion:
        call_kwargs = self.parent._merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        start = time.perf_counter()
        response = self.parent._client.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        response_dict = response.model_dump()

        completion_token_ids = self.parent._extract_completion_tokens(response_dict)

        self.parent._log_trace(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_payload=response_dict,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=latency_ms,
        )

        return response


class SimpleTrackedChatClient(_SimpleTrackedChatClientBase):
    """Lean wrapper around `OpenAI` that records chat completions only."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        tracer: Any = None,
        default_model: str | None = None,
        client: OpenAI | None = None,
        **client_kwargs: Any,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            init_kwargs = dict(client_kwargs)
            if api_key is not None:
                init_kwargs["api_key"] = api_key
            if base_url is not None:
                init_kwargs["base_url"] = base_url
            self._client = OpenAI(**init_kwargs)

        self.tracer = tracer
        self.default_model = default_model

        self.chat = _ChatNamespace(self)
        self.completions = _Completions(self)


@dataclass
class _AsyncChatCompletions:
    parent: SimpleTrackedAsyncChatClient

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = self.parent._merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        start = time.perf_counter()
        response = await self.parent._client.chat.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        response_dict = response.model_dump()

        completion_token_ids = self.parent._extract_completion_tokens(response_dict)

        self.parent._log_trace(
            model=model,
            messages=messages,
            response_payload=response_dict,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=latency_ms,
        )

        return response


@dataclass
class _AsyncChatNamespace:
    parent: SimpleTrackedAsyncChatClient

    @property
    def completions(self) -> _AsyncChatCompletions:
        return _AsyncChatCompletions(self.parent)


@dataclass
class _AsyncCompletions:
    parent: SimpleTrackedAsyncChatClient

    async def create(self, *args: Any, **kwargs: Any) -> Completion:
        call_kwargs = self.parent._merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        start = time.perf_counter()
        response = await self.parent._client.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        response_dict = response.model_dump()

        completion_token_ids = self.parent._extract_completion_tokens(response_dict)

        self.parent._log_trace(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_payload=response_dict,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=latency_ms,
        )

        return response


class SimpleTrackedAsyncChatClient(_SimpleTrackedChatClientBase):
    """Async variant of the simple client that records chat completions."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        tracer: Any = None,
        default_model: str | None = None,
        client: AsyncOpenAI | None = None,
        **client_kwargs: Any,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            init_kwargs = dict(client_kwargs)
            if api_key is not None:
                init_kwargs["api_key"] = api_key
            if base_url is not None:
                init_kwargs["base_url"] = base_url
            self._client = AsyncOpenAI(**init_kwargs)

        self.tracer = tracer
        self.default_model = default_model

        self.chat = _AsyncChatNamespace(self)
        self.completions = _AsyncCompletions(self)

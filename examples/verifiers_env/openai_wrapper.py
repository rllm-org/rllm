"""Wrapper that makes RolloutEngine look like AsyncOpenAI for verifiers compatibility."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs, CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob

if TYPE_CHECKING:
    from rllm.engine.rollout.rollout_engine import RolloutEngine


class ChatCompletions:
    """Implements client.chat.completions interface."""

    def __init__(
        self,
        engine: RolloutEngine,
        model: str,
        application_id_fn: Callable[[], str] | None = None,
    ):
        self._engine = engine
        self._model = model
        self._application_id_fn = application_id_fn or (lambda: "default")

    async def create(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ChatCompletion:
        """Call RolloutEngine and return OpenAI-compatible ChatCompletion."""
        # Normalize sampling args
        sampling_args = dict(kwargs)

        # max_completion_tokens → max_tokens for engine
        if "max_completion_tokens" in sampling_args:
            sampling_args["max_tokens"] = sampling_args.pop("max_completion_tokens")

        # Remove args the engine doesn't understand
        sampling_args.pop("extra_body", None)
        sampling_args.pop("modalities", None)

        # Call the engine
        output = await self._engine.get_model_response(
            messages=messages,
            application_id=self._application_id_fn(),
            tools=tools or [],
            **sampling_args,
        )

        # Convert tool_calls if present
        # tool_calls can be:
        # - list[ToolCall] (rllm dataclass with name, arguments)
        # - list[ChatCompletionMessageToolCall] (raw OpenAI objects)
        # - list[dict] (parsed dicts)
        tool_calls = None
        if output.tool_calls:
            converted_tool_calls = []
            for i, tc in enumerate(output.tool_calls):
                # Already an OpenAI ChatCompletionMessageToolCall - pass through
                if isinstance(tc, ChatCompletionMessageToolCall):
                    converted_tool_calls.append(tc)
                # rllm ToolCall dataclass or dict
                else:
                    tc_id = getattr(tc, "id", None) or f"call_{i}"
                    tc_name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else "")
                    tc_args = getattr(tc, "arguments", None) or (tc.get("arguments") if isinstance(tc, dict) else {})
                    # arguments can be dict or string
                    if isinstance(tc_args, dict):
                        tc_args = json.dumps(tc_args)
                    converted_tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=tc_id,
                            type="function",
                            function=Function(name=tc_name, arguments=tc_args),
                        )
                    )
            tool_calls = converted_tool_calls if converted_tool_calls else None

        # Build logprobs if available
        logprobs = None
        if output.logprobs:
            token_logprobs = [
                ChatCompletionTokenLogprob(
                    token=f"<token_{i}>",  # placeholder, we don't have decoded tokens
                    bytes=None,
                    logprob=float(lp) if lp is not None else 0.0,
                    top_logprobs=[],
                )
                for i, lp in enumerate(output.logprobs)
            ]
            logprobs = ChoiceLogprobs(content=token_logprobs, refusal=None)

        # Build the choice
        choice = Choice(
            index=0,
            message=ChatCompletionMessage(
                role="assistant",
                content=output.content or output.text,
                tool_calls=tool_calls,
            ),
            finish_reason=output.finish_reason or "stop",
            logprobs=logprobs,
        )

        # Add vLLM extensions as attributes
        choice.token_ids = output.completion_ids  # type: ignore[attr-defined]

        # Build the response
        response = ChatCompletion(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            model=model or self._model,
            object="chat.completion",
            created=int(time.time()),
            choices=[choice],
            usage=CompletionUsage(
                prompt_tokens=output.prompt_length or 0,
                completion_tokens=output.completion_length or 0,
                total_tokens=(output.prompt_length or 0) + (output.completion_length or 0),
            ),
        )

        # Add vLLM extension
        response.prompt_token_ids = output.prompt_ids  # type: ignore[attr-defined]

        return response


class Chat:
    """Implements client.chat interface."""

    def __init__(
        self,
        engine: RolloutEngine,
        model: str,
        application_id_fn: Callable[[], str] | None = None,
    ):
        self.completions = ChatCompletions(engine, model, application_id_fn)


class RolloutEngineAsyncClient:
    """
    Wrapper that makes RolloutEngine look like AsyncOpenAI.

    Use this to integrate rllm with libraries that expect an AsyncOpenAI client,
    such as the verifiers library.

    Example:
        client = RolloutEngineAsyncClient(
            rollout_engine=self.rollout_engine,
            model="Qwen/Qwen3-0.6B",
            application_id_fn=lambda: uid,
        )

        # Now use with verifiers
        response = await client.chat.completions.create(
            model="...",
            messages=[...],
        )
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        model: str = "",
        application_id_fn: Callable[[], str] | None = None,
        base_url: str = "http://rllm-internal",
    ):
        self._engine = rollout_engine
        self._model = model or getattr(rollout_engine, "model", "unknown")
        self._application_id_fn = application_id_fn

        # Properties that verifiers accesses
        self.base_url = base_url
        self.api_key = "rllm-internal"

        # Build interface
        self.chat = Chat(rollout_engine, self._model, application_id_fn)

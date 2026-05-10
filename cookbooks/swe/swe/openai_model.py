#!/usr/bin/env python3
"""OpenAI-client-backed model for mini-swe-agent v2.

Pure protocol translation: wraps ``openai.OpenAI(base_url=...)`` to
implement mini-swe-agent's ``Model`` interface (``query``,
``format_message``, ``format_observation_messages``, ``get_template_vars``,
``serialize``).

Uses native tool calling (``tools=[BASH_TOOL]``) matching mini-swe-agent
v2's default LitellmModel behavior.

No training plumbing — logprobs are captured externally by the rllm
model gateway.
"""

from __future__ import annotations

import time
from typing import Any

from jinja2 import Template
from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ConfigDict, Field

from swe.utils import (
    is_context_length_error,
    tool_response_user_message,
)

# mini-swe-agent imports are deferred to avoid a circular import:
#   openai_model → swe.environment → agent_flow → openai_model
_BASH_TOOL = None
_parse_toolcall_actions = None
_format_toolcall_observation_messages = None
_FormatError = None

_API_MESSAGE_KEYS = {"role", "content", "tool_calls", "tool_call_id", "name"}
_REQUEST_KWARGS = {
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "presence_penalty",
    "seed",
    "stop",
    "temperature",
    "tool_choice",
    "top_logprobs",
    "top_p",
}
_TOOL_REQUEST_KWARGS = {"parallel_tool_calls", "tool_choice"}
_EXTRA_BODY_KEYS = ("chat_template_kwargs", "guided_decoding")


class MaxPromptLengthExceeded(RuntimeError):
    """Raised when the request prompt/context exceeds the model limit."""


class MaxResponseLengthExceeded(RuntimeError):
    """Raised when generation stops because the response token limit was reached."""


def _extract_compaction_summary(text: str) -> str:
    """Strip ``<think>...</think>`` and any ``<tool_call>...`` from a
    compaction completion.

    The directive tells the model not to call tools, but Qwen3.5 sometimes
    ignores it and continues with a tool call after closing thinking. With
    the previous ``text.split("</think>")[-1]`` extractor that whole
    ``<tool_call>...</tool_call>`` block became the "summary" and was
    injected into the post-compaction conversation as a user message —
    contaminating downstream training and rendering.
    """
    body = text.split("</think>")[-1]
    body = body.split("<tool_call>", 1)[0]
    return body.strip()


def _ensure_minisweagent_imports():
    global _BASH_TOOL, _parse_toolcall_actions, _format_toolcall_observation_messages, _FormatError
    if _BASH_TOOL is not None:
        return
    from minisweagent.models.utils.actions_toolcall import (
        BASH_TOOL,
        format_toolcall_observation_messages,
        parse_toolcall_actions,
    )
    from minisweagent.exceptions import FormatError
    _BASH_TOOL = BASH_TOOL
    _parse_toolcall_actions = parse_toolcall_actions
    _format_toolcall_observation_messages = format_toolcall_observation_messages
    _FormatError = FormatError


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

class OpenAIClientModelConfig(BaseModel):
    """Fields that mini-swe-agent Jinja templates may reference."""

    model_config = ConfigDict(extra="ignore")

    model_name: str = "unknown"
    tokenizer_name: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    format_error_template: str = "{{ error }}"
    compaction_continuation_template: str = "{{ summary }}"
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------

class OpenAIClientModel:
    """mini-swe-agent v2 model backed by any OpenAI-compatible endpoint.

    Uses native tool calling with the bash tool, matching mini-swe-agent
    v2's default LitellmModel behavior.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        model_config: dict | None = None,
        verbose: bool = False,
    ):
        cfg = dict(model_config or {})
        cfg.pop("action_regex", None)  # legacy text-based; no longer used
        cfg.setdefault("model_name", model_name)
        self.config = OpenAIClientModelConfig(**cfg)
        self.verbose = verbose

        client_kwargs: dict[str, Any] = {"base_url": base_url}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self.client = OpenAI(**client_kwargs)

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._official_openai_api = self.base_url.lower() in {
            "https://api.openai.com",
            "https://api.openai.com/v1",
        }
        self._allow_extra_body = not self._official_openai_api
        self.tokenizer = None

        self.cost = 0.0
        self.n_calls = 0

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[OpenAIClientModel] {msg}")

    def _request_kwargs(self, *, include_tools: bool) -> dict[str, Any]:
        model_kwargs = self.config.model_kwargs
        kwargs = {}
        for key, value in model_kwargs.items():
            if key not in _REQUEST_KWARGS or value is None:
                continue
            if not include_tools and key in _TOOL_REQUEST_KWARGS:
                continue
            if self._official_openai_api and self.model_name.lower().startswith("gpt-5") and key == "temperature":
                continue
            kwargs[key] = value

        if include_tools:
            kwargs["parallel_tool_calls"] = model_kwargs.get("parallel_tool_calls", False)

        if self._allow_extra_body:
            extra_body = dict(model_kwargs.get("extra_body") or {})
            for key in _EXTRA_BODY_KEYS:
                if key in model_kwargs:
                    extra_body.setdefault(key, model_kwargs[key])
            if "return_token_ids" in model_kwargs:
                extra_body.setdefault("return_token_ids", bool(model_kwargs["return_token_ids"]))
            if extra_body:
                kwargs["extra_body"] = extra_body

        return kwargs

    @staticmethod
    def _is_request_parameter_error(error_msg: str) -> bool:
        msg = error_msg.lower()
        return "unknown parameter" in msg or "unsupported value" in msg

    @staticmethod
    def _api_message(message: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in message.items()
            if key in _API_MESSAGE_KEYS and value is not None
        }

    @staticmethod
    def _completion_token_ids(response_dump: dict[str, Any]) -> list[int] | None:
        choices = response_dump.get("choices") or []
        if not choices:
            return None
        token_ids = choices[0].get("token_ids")
        if token_ids is None:
            return None
        return list(token_ids)

    def _get_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            name = self.config.tokenizer_name or self.model_name
            tokenizer = AutoTokenizer.from_pretrained(
                name,
                trust_remote_code=True,
                local_files_only=True,
            )
            if tokenizer.eos_token_id is None:
                raise RuntimeError(
                    f"Tokenizer for {name} has no eos_token_id; cannot use token-aware mode"
                )
            self.tokenizer = tokenizer
        return self.tokenizer

    def _decode_completion(self, token_ids: list[int]) -> str:
        """Decode sampled tokens, splitting off a trailing EOS if present.

        Does not raise on missing EOS. The caller treats
        ``finish_reason == "length"`` as the canonical truncation signal;
        with vLLM's qwen3_coder tool-call parser, generation can stop at
        ``</tool_call>`` without emitting ``<|im_end|>``, so requiring EOS
        here would false-positive on normal completions.
        """
        tokenizer = self._get_tokenizer()
        eos_id = tokenizer.eos_token_id
        if eos_id in token_ids:
            token_ids = token_ids[:token_ids.index(eos_id)]
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    # ---- mini-swe-agent Model protocol ---------------------------------

    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict:
        """Call the model and return an assistant message with tool calls.

        Uses native tool calling. Format errors are signalled via
        ``extra["format_error"]`` instead of raising.
        """
        _ensure_minisweagent_imports()
        self._log(f"Query #{self.n_calls + 1}: {len(messages)} messages")

        api_messages = [self._api_message(msg) for msg in messages]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                tools=[_BASH_TOOL],
                **self._request_kwargs(include_tools=True),
            )
        except BadRequestError as e:
            error_msg = str(e)
            if is_context_length_error(error_msg):
                raise MaxPromptLengthExceeded() from e
            if self._is_request_parameter_error(error_msg):
                raise
            self.n_calls += 1
            return self._bad_request_as_format_error(error_msg)

        response_dump = response.model_dump()
        choice = response.choices[0]
        message = choice.message.model_dump(exclude_none=True)
        finish_reason = choice.finish_reason or "stop"
        self.n_calls += 1
        completion_token_ids = self._completion_token_ids(response_dump)
        raw_transcript = completion_token_ids is not None

        if finish_reason == "length":
            raise MaxResponseLengthExceeded()
        if raw_transcript:
            self._log("Using completion token ids for model response")
            message["content"] = self._decode_completion(completion_token_ids)
            # Only mutate the transcript dict. The parser below still reads
            # the original Pydantic choice.message.tool_calls.
            message.pop("tool_calls", None)
        else:
            self._log("Using text based model response")

        format_error = None
        try:
            tool_calls = choice.message.tool_calls or []
            actions = _parse_toolcall_actions(
                tool_calls,
                format_error_template=self.config.format_error_template,
            )
        except _FormatError as e:
            actions = []
            format_error = e.messages[0]["content"]
            message.pop("tool_calls", None)
            message["content"] = message.get("content") or ""

        text = message.get("content") or ""
        self._log(
            f"Response: {len(text)} chars, finish={finish_reason}"
            + (", format_error" if format_error else f", {len(actions)} tool calls")
        )

        extra: dict[str, Any] = {
            "actions": actions,
            "cost": 0.0,
            "response": response_dump,
            "timestamp": time.time(),
            "raw_transcript": raw_transcript,
        }
        if completion_token_ids is not None:
            extra["completion_token_ids"] = completion_token_ids
        if format_error is not None:
            extra["format_error"] = format_error

        message["extra"] = extra
        return message

    def _bad_request_as_format_error(self, error_msg: str) -> dict:
        """Wrap a malformed-tool-call 400 as an assistant message with
        ``extra["format_error"]`` so the agent can retry. Context-length
        overflows must be re-raised by the caller, not routed here.
        """
        self._log(f"API error converted to format error: {error_msg[:200]}")
        format_error_content = Template(self.config.format_error_template).render(
            error=f"Your tool call could not be parsed: {error_msg[:300]}"
        )
        return {
            "role": "assistant",
            "content": "",
            "extra": {
                "format_error": format_error_content,
                "actions": [],
                "cost": 0.0,
                "timestamp": time.time(),
                "raw_transcript": False,
            },
        }

    def summarize_context(
        self,
        messages: list[dict[str, Any]],
        summary_prompt: str,
    ) -> dict[str, Any]:
        """Call the model to summarize conversation history.

        Returns a synthetic user message with extra["summary"] = True. The
        gateway captures the model's raw summarizer completion separately.
        """
        self._log(f"Summarize context: {len(messages)} messages")

        # Wrap the directive in <tool_response> so Qwen3.5's reverse scan
        # for last_query_index skips it, preserving <think> blocks on every
        # prior assistant. A bare user message here strips thinking from all
        # past turns (same failure mode as the format_error path).
        directive_message = tool_response_user_message(summary_prompt)
        summary_messages = [
            self._api_message(msg) for msg in messages
        ] + [directive_message]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=summary_messages,
                tools=[_BASH_TOOL],
                **self._request_kwargs(include_tools=True),
            )
        except BadRequestError as e:
            error_msg = str(e)
            if is_context_length_error(error_msg):
                raise MaxPromptLengthExceeded() from e
            raise
        response_dump = response.model_dump()
        choice = response.choices[0]
        self.n_calls += 1

        if choice.finish_reason == "length":
            raise MaxResponseLengthExceeded()

        completion_token_ids = self._completion_token_ids(response_dump)
        raw_transcript = completion_token_ids is not None
        if raw_transcript:
            text = self._decode_completion(completion_token_ids)
        else:
            text = choice.message.content or ""

        summary = _extract_compaction_summary(text)
        content = Template(self.config.compaction_continuation_template).render(summary=summary)
        self._log(f"Summary response: {len(text)} chars, compacted to {len(summary)} chars")

        return {
            "role": "user",
            "content": content,
            "extra": {
                "summary": True,
                "cost": 0.0,
                "response": response_dump,
                "timestamp": time.time(),
                "raw_transcript": raw_transcript,
            },
        }

    def format_message(self, **kwargs) -> dict:
        return dict(kwargs)

    def format_observation_messages(self, message, outputs, template_vars=None):
        _ensure_minisweagent_imports()
        actions = message.get("extra", {}).get("actions", [])
        messages = _format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
        )
        if not message.get("extra", {}).get("raw_transcript"):
            return messages

        raw_messages = []
        for msg in messages:
            if msg.get("role") == "tool":
                raw_messages.append(tool_response_user_message(str(msg.get("content", ""))))
            else:
                raw_messages.append(msg)
        return raw_messages

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump() | {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
        }

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            },
        }

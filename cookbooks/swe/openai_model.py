#!/usr/bin/env python3
"""OpenAI-client-backed model for mini-swe-agent v2.

Pure protocol translation: wraps ``openai.OpenAI(base_url=...)`` to
implement mini-swe-agent's ``Model`` interface (``query``,
``format_message``, ``format_observation_messages``, ``get_template_vars``,
``serialize``).

No training plumbing — logprobs are captured externally by the rllm
model gateway.
"""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict

# mini-swe-agent imports are deferred to avoid a circular import:
#   openai_model → swe.environment → agent_flow → openai_model
_parse_regex_actions = None
_format_text_observation_messages = None
_FormatError = None


def _ensure_minisweagent_imports():
    global _parse_regex_actions, _format_text_observation_messages, _FormatError
    if _parse_regex_actions is not None:
        return
    from minisweagent.models.utils.actions_text import (
        format_observation_messages as fmt,
        parse_regex_actions as pra,
    )
    from minisweagent.exceptions import FormatError
    _parse_regex_actions = pra
    _format_text_observation_messages = fmt
    _FormatError = FormatError


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

class OpenAIClientModelConfig(BaseModel):
    """Fields that mini-swe-agent Jinja templates may reference."""

    model_config = ConfigDict(extra="ignore")

    model_name: str = "unknown"
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, "
        "found {{actions|length}} actions."
    )
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------

class OpenAIClientModel:
    """mini-swe-agent v2 model backed by any OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        model_config: dict | None = None,
        verbose: bool = False,
    ):
        cfg = dict(model_config or {})
        cfg.pop("model_kwargs", None)  # provider-specific; not forwarded
        cfg.setdefault("model_name", model_name)
        self.config = OpenAIClientModelConfig(**cfg)

        client_kwargs: dict[str, Any] = {"base_url": base_url}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self.client = OpenAI(**client_kwargs)

        self.model_name = model_name
        self.verbose = verbose
        self.cost = 0.0
        self.n_calls = 0

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[OpenAIClientModel] {msg}")

    # ---- mini-swe-agent Model protocol ---------------------------------

    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict:
        """Call the model and return an assistant message.

        Always returns ``role: "assistant"`` — format errors are
        signalled via ``extra["format_error"]`` instead of raising.
        """
        _ensure_minisweagent_imports()
        self._log(f"Query #{self.n_calls + 1}: {len(messages)} messages")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        choice = response.choices[0]
        text = choice.message.content or ""
        finish_reason = choice.finish_reason or "stop"
        self.n_calls += 1

        format_error = None
        try:
            actions = _parse_regex_actions(
                text,
                action_regex=self.config.action_regex,
                format_error_template=self.config.format_error_template,
            )
        except _FormatError as e:
            actions = []
            format_error = e.messages[0]["content"]

        self._log(f"Response: {len(text)} chars, finish={finish_reason}"
                   + (", format_error" if format_error else ""))

        extra: dict[str, Any] = {
            "actions": actions,
            "cost": 0.0,
            "response": response.model_dump(),
            "timestamp": time.time(),
        }
        if format_error is not None:
            extra["format_error"] = format_error

        return {"role": "assistant", "content": text, "extra": extra}

    def format_message(self, **kwargs) -> dict:
        return dict(kwargs)

    def format_observation_messages(self, message, outputs, template_vars=None):
        _ensure_minisweagent_imports()
        return _format_text_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
        )

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

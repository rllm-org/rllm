import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class APIProtocol(str, Enum):
    COMPLETIONS = "completions"
    CHAT_COMPLETIONS = "chat_completions"


class CanonicalRequest(BaseModel):
    request_id: str
    session_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    prompt: str | None = None
    prompt_token_ids: list[int] | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    sampling_params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_one_input(self) -> "CanonicalRequest":
        input_count = int(bool(self.messages)) + int(self.prompt is not None) + int(self.prompt_token_ids is not None)
        if input_count != 1:
            raise ValueError("canonical request must contain exactly one of messages, prompt, or prompt_token_ids")
        return self


class CanonicalOutput(BaseModel):
    request_id: str
    text: str
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int
    completion_tokens: int


class TokenInput(BaseModel):
    session_id: str
    prompt_token_ids: list[int]
    sampling_params: dict[str, Any] = Field(default_factory=dict)


class TokenOutput(BaseModel):
    completion_token_ids: list[int]
    logprobs: list[float] | None = None
    routed_experts: list[str] | None = None
    finish_reason: str | None = None
    weight_version: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    request_id: str
    input: TokenInput
    output: TokenOutput
    created_at: float = Field(default_factory=time.time)


class SessionTraces(BaseModel):
    session_id: str
    traces: list[Trace] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)

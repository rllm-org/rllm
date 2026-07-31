import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class GatewayError(Exception):
    def __init__(self, message: str, status_code: int = 400, error_type: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type


class WorkerUnavailableError(GatewayError):
    def __init__(self, worker_id: int) -> None:
        super().__init__(f"gateway worker {worker_id} is unavailable", 503, "server_error")


class APIProtocol(str, Enum):
    COMPLETIONS = "completions"
    CHAT_COMPLETIONS = "chat_completions"


class GatewayRequest(BaseModel):
    request_id: str
    session_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    prompt: str | None = None
    prompt_token_ids: list[int] | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    sampling_params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_one_input(self) -> "GatewayRequest":
        input_count = int(bool(self.messages)) + int(self.prompt is not None) + int(self.prompt_token_ids is not None)
        if input_count != 1:
            raise ValueError("gateway request must contain exactly one of messages, prompt, or prompt_token_ids")
        return self


class GatewayResponse(BaseModel):
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


class TraceRecord(BaseModel):
    request: GatewayRequest
    response: GatewayResponse
    input: TokenInput
    output: TokenOutput
    started_at: float
    completed_at: float


class SessionTraces(BaseModel):
    session_id: str
    traces: list[TraceRecord] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)

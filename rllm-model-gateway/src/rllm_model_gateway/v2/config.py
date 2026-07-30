from typing import Any

from pydantic import BaseModel, Field, model_validator


class TokenizationConfig(BaseModel):
    model: str
    trust_remote_code: bool = False
    renderer: str = "auto"
    renderer_kwargs: dict[str, Any] = Field(default_factory=dict)


class BackendConfig(BaseModel):
    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class WorkerProcessConfig(BaseModel):
    cumulative: bool = False
    tokenization: TokenizationConfig
    backend: BackendConfig


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 9090
    num_workers: int = Field(default=1, ge=1)
    request_timeout_seconds: float = Field(default=3600.0, gt=0)
    heartbeat_seconds: float = Field(default=10.0, gt=0)
    admin_key: str
    agent_base_url: str
    cumulative: bool = False
    tokenization: TokenizationConfig
    backend: BackendConfig

    @model_validator(mode="after")
    def validate_required_fields(self) -> "GatewayConfig":
        if not self.admin_key:
            raise ValueError("admin_key must not be empty")
        if not self.agent_base_url:
            raise ValueError("agent_base_url must not be empty")
        return self

    def worker_config(self) -> WorkerProcessConfig:
        return WorkerProcessConfig(cumulative=self.cumulative, tokenization=self.tokenization, backend=self.backend)

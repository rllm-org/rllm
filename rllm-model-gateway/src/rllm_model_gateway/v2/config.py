from typing import Any

from pydantic import BaseModel, Field, model_validator


class WorkerProcessConfig(BaseModel):
    cumulative: bool = False
    tokenizer_model: str
    renderer: str = "auto"
    renderer_kwargs: dict[str, Any] = Field(default_factory=dict)


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 9090
    num_workers: int = Field(default=1, ge=1)
    worker_startup_timeout_seconds: float = Field(default=300.0, gt=0)
    request_timeout_seconds: float = Field(default=3600.0, gt=0)
    update_timeout_seconds: float = Field(default=300.0, gt=0)
    heartbeat_initial_delay_seconds: float = Field(default=60.0, gt=0)
    heartbeat_interval_seconds: float = Field(default=10.0, gt=0)
    admin_key: str
    cumulative: bool = False
    tokenizer_model: str
    renderer: str = "auto"
    renderer_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_required_fields(self) -> "GatewayConfig":
        if not self.admin_key:
            raise ValueError("admin_key must not be empty")
        return self

    def worker_config(self) -> WorkerProcessConfig:
        return WorkerProcessConfig(
            cumulative=self.cumulative,
            tokenizer_model=self.tokenizer_model,
            renderer=self.renderer,
            renderer_kwargs=self.renderer_kwargs,
        )

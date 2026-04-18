"""rllm-model-gateway: lightweight LLM call gateway for RL agent training and eval."""

from rllm_model_gateway._version import __version__
from rllm_model_gateway.client import AsyncGatewayClient, GatewayClient
from rllm_model_gateway.config import GatewayConfig
from rllm_model_gateway.normalized import (
    AdapterError,
    Message,
    NormalizedRequest,
    NormalizedResponse,
    ToolCall,
    ToolSpec,
    Usage,
)
from rllm_model_gateway.server import AdapterFn, create_app
from rllm_model_gateway.trace import TraceRecord, build_trace, deserialize_extras, serialize_extras

__all__ = [
    "__version__",
    "create_app",
    "GatewayConfig",
    "AdapterFn",
    "AdapterError",
    "GatewayClient",
    "AsyncGatewayClient",
    "NormalizedRequest",
    "NormalizedResponse",
    "Message",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "TraceRecord",
    "build_trace",
    "serialize_extras",
    "deserialize_extras",
]

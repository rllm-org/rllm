"""Rllm Telemetry — Agent observability plugin for Google ADK."""

from .config import RllmConfig
from .exporter import BaseExporter, HttpExporter, StdoutExporter, create_exporter
from .plugin import RllmTelemetryPlugin
from .schemas import (
    AgentInfo,
    AgentSpanRecord,
    EventActionsData,
    EventRecord,
    GenerationConfig,
    InvocationRecord,
    LlmRequest,
    LlmResponseData,
    LlmSpanRecord,
    SessionRecord,
    ToolInfo,
    ToolSpanRecord,
    TraceEnvelope,
    UsageMetadata,
)

__all__ = [
    # Plugin (primary API)
    "RllmTelemetryPlugin",
    "RllmConfig",
    # Exporters
    "BaseExporter",
    "HttpExporter",
    "StdoutExporter",
    "create_exporter",
    # Convenience
    "instrument",
    # Schemas (for custom backends / testing)
    "AgentInfo",
    "AgentSpanRecord",
    "EventActionsData",
    "EventRecord",
    "GenerationConfig",
    "InvocationRecord",
    "LlmRequest",
    "LlmResponseData",
    "LlmSpanRecord",
    "SessionRecord",
    "ToolInfo",
    "ToolSpanRecord",
    "TraceEnvelope",
    "UsageMetadata",
]


def instrument(runner, *, api_key: str = "", endpoint: str = "", **kwargs):
    """One-liner convenience to attach Rllm telemetry to a Runner.

    Example::

        import rllm_telemetry
        rllm_telemetry.instrument(runner, api_key="sk-...")
    """
    config_kwargs = {"api_key": api_key, **kwargs}
    if endpoint:
        config_kwargs["endpoint"] = endpoint
    config = RllmConfig(**config_kwargs)
    plugin = RllmTelemetryPlugin(config=config)
    runner.plugin_manager.register_plugin(plugin)
    return plugin

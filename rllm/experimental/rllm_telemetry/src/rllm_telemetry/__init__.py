"""Rllm Telemetry — Agent observability plugin for Google ADK."""

from .compare import ComparisonResult, compare
from .config import RllmConfig
from .dataset import load_dataset
from .eval import AsyncEval, Eval, ExperimentResult
from .exporter import AgentSpanExporter, AgentTrajectoryExporter, BaseExporter, BigQueryExporter, BigQueryValidationError, HttpExporter, StdoutExporter, create_exporter, get_bq_table_schema
from .plugin import RllmTelemetryPlugin
from .schemas import (
    AgentInfo,
    AgentSpanRecord,
    EventActionsData,
    EventRecord,
    ExperimentCaseRecord,
    ExperimentRecord,
    ExperimentSummary,
    GenerationConfig,
    InvocationRecord,
    LlmRequest,
    LlmResponseData,
    LlmSpanRecord,
    ScoreRecord,
    SessionRecord,
    SessionStartRecord,
    ToolDataRecord,
    ToolInfo,
    ToolSpanRecord,
    TraceEnvelope,
    UsageMetadata,
)
from .scorers import JUDGE_PROMPTS, Contains, ExactMatch, LlmJudge, Score, Scorer
from .trajectory_export import export_trajectories

__all__ = [
    # Plugin (primary API)
    "RllmTelemetryPlugin",
    "RllmConfig",
    # Exporters
    "AgentSpanExporter",
    "AgentTrajectoryExporter",
    "BaseExporter",
    "BigQueryExporter",
    "BigQueryValidationError",
    "HttpExporter",
    "StdoutExporter",
    "create_exporter",
    "get_bq_table_schema",
    # Scorers
    "Score",
    "Scorer",
    "ExactMatch",
    "Contains",
    "LlmJudge",
    "JUDGE_PROMPTS",
    # Experiments
    "Eval",
    "AsyncEval",
    "ExperimentResult",
    # Comparison
    "compare",
    "ComparisonResult",
    # Dataset loading
    "load_dataset",
    # Trajectory export
    "export_trajectories",
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
    "SessionStartRecord",
    "ToolDataRecord",
    "ToolInfo",
    "ToolSpanRecord",
    "ExperimentRecord",
    "ExperimentCaseRecord",
    "ExperimentSummary",
    "ScoreRecord",
    "TraceEnvelope",
    "UsageMetadata",
]


def instrument(
    runner,
    *,
    api_key: str = "",
    endpoint: str = "",
    agent_endpoint: str = "",
    agent_session_name: str = "",
    **kwargs,
):
    """One-liner convenience to attach Rllm telemetry to a Runner.

    Example::

        import rllm_telemetry

        # Basic stdout tracing
        rllm_telemetry.instrument(runner, api_key="sk-...")

        # With real-time trajectory streaming to rllm_ui
        rllm_telemetry.instrument(
            runner,
            agent_endpoint="http://localhost:8000",
            backend="stdout",
        )
    """
    config_kwargs = {"api_key": api_key, **kwargs}
    if endpoint:
        config_kwargs["endpoint"] = endpoint
    if agent_endpoint:
        config_kwargs["agent_endpoint"] = agent_endpoint
    if agent_session_name:
        config_kwargs["agent_session_name"] = agent_session_name
    config = RllmConfig(**config_kwargs)
    plugin = RllmTelemetryPlugin(config=config)
    runner.plugin_manager.register_plugin(plugin)
    return plugin

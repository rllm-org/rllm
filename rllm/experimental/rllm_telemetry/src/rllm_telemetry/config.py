"""Configuration for the Rllm telemetry plugin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RllmConfig:
    """Configuration for :class:`RllmTelemetryPlugin`.

    Attributes:
        backend: Exporter backend to use.  ``"http"`` sends traces to a
            remote endpoint; ``"stdout"`` pretty-prints them to stdout
            for local debugging.  Defaults to ``"http"``.
        api_key: API key for authenticating with the Rllm backend.
        endpoint: Base URL of the Rllm ingest API.
        capture_content: Whether to capture full LLM request/response content
            (system prompt, chat history, model output).  Set to ``False`` to
            redact prompt content for privacy.  Tool args/results and event
            content follow the same flag.  Defaults to ``True``.
        capture_tools: Whether to capture tool args and results.
            Defaults to ``True``.
        max_content_length: Maximum character length for serialised content
            fields before truncation.  ``-1`` disables truncation.
        batch_size: Number of records to buffer before flushing to the
            backend.
        flush_interval_seconds: Maximum seconds between flushes.
        headers: Extra HTTP headers to include on every request.
        timeout_seconds: HTTP request timeout.
    """

    backend: Literal["http", "stdout", "bigquery"] = "stdout"
    api_key: str = ""
    endpoint: str = "http://localhost:8100/v1/traces"
    capture_content: bool = True
    capture_tools: bool = True
    max_content_length: int = -1
    batch_size: int = 64
    flush_interval_seconds: float = 2.0
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 10.0
    color: bool = True
    """Enable ANSI color output for the ``stdout`` backend.  Ignored by
    other backends."""

    # --- Agent trajectory streaming (to rllm_ui backend) ---
    agent_endpoint: str = ""
    """Base URL of the rllm_ui backend for real-time agent trajectory
    streaming.  When set, trajectory spans are sent here per-record.
    Example: ``"http://localhost:8000"``."""

    agent_api_key: str = ""
    """API key for authenticating with the rllm_ui agent endpoint.
    Falls back to ``api_key`` if not set."""

    agent_session_name: str = ""
    """Human-readable name for the agent session.  Auto-generated if
    empty."""

    # --- BigQuery backend ---
    bq_project: str = ""
    """Google Cloud project ID for BigQuery."""

    bq_dataset: str = ""
    """BigQuery dataset name."""

    bq_table: str = "rllm_traces"
    """BigQuery table name within the dataset."""

    bq_auto_create: bool = False
    """Automatically create the BigQuery dataset and table if they don't
    exist.  When ``False`` (the default), :meth:`start` raises
    :class:`BigQueryValidationError` if either is missing."""

"""Exporters that ship trace records to various backends.

The :class:`BaseExporter` protocol defines the interface.  Four concrete
implementations are provided:

* :class:`HttpExporter` — async batching HTTP exporter (NDJSON).
* :class:`StdoutExporter` — pretty-prints traces to stdout for local testing.
* :class:`BigQueryExporter` — async batching exporter to Google BigQuery.
* :class:`AgentSpanExporter` — composite exporter that routes all span
  events to the rllm_ui backend in real-time (per-record).
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import sys
import uuid
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .config import RllmConfig
from .schemas import SpanType, TraceEnvelope

logger = logging.getLogger("rllm_telemetry.exporter")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseExporter(ABC):
    """Interface that all trace exporters must implement."""

    @abstractmethod
    def enqueue(self, span_type: SpanType, data: dict[str, Any]) -> None:
        """Buffer a trace record for export."""

    @abstractmethod
    async def start(self) -> None:
        """Initialise resources (called once before the first enqueue)."""

    @abstractmethod
    async def close(self) -> None:
        """Flush remaining records and release resources."""


# ---------------------------------------------------------------------------
# Stdout exporter
# ---------------------------------------------------------------------------


# ANSI color codes for stdout exporter
_COLORS: dict[str, str] = {
    "session": "\033[36m",  # cyan
    "session.start": "\033[36m",  # cyan
    "invocation.start": "\033[34m",  # blue
    "invocation.end": "\033[34m",
    "agent.start": "\033[35m",  # magenta
    "agent.end": "\033[35m",
    "llm.start": "\033[33m",  # yellow
    "llm.end": "\033[33m",
    "tool.start": "\033[32m",  # green
    "tool.end": "\033[32m",
    "tool.data": "\033[32m",  # green
    "event": "\033[37m",  # white/gray
    "experiment.start": "\033[96m",  # bright cyan
    "experiment.end": "\033[96m",
    "experiment.case": "\033[94m",  # bright blue
}
_RESET = "\033[0m"


class StdoutExporter(BaseExporter):
    """Exporter that pretty-prints trace envelopes to stdout.

    Useful for local development and debugging — no backend required.

    When ``color=True`` (the default), output is color-coded by span type:

    - **Cyan**: session
    - **Blue**: invocation
    - **Magenta**: agent spans
    - **Yellow**: LLM calls
    - **Green**: tool executions
    - **Gray**: events
    - **Bright cyan/blue**: experiment records
    """

    def __init__(self, config: RllmConfig) -> None:
        self._config = config
        self._closed = False
        self._color = config.color

    def enqueue(self, span_type: SpanType, data: dict[str, Any]) -> None:
        if self._closed:
            return
        envelope = TraceEnvelope(type=span_type, data=data)
        payload = json.loads(envelope.model_dump_json(exclude_none=True))
        text = json.dumps(payload, indent=2)
        if self._color:
            color = _COLORS.get(span_type, "")
            text = f"{color}{text}{_RESET}"
        print(text, file=sys.stdout, flush=True)

    async def start(self) -> None:
        self._closed = False

    async def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# HTTP exporter
# ---------------------------------------------------------------------------


class HttpExporter(BaseExporter):
    """Async batching HTTP exporter (NDJSON wire format)."""

    def __init__(self, config: RllmConfig) -> None:
        self._config = config
        self._buffer: list[TraceEnvelope] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, span_type: SpanType, data: dict[str, Any]) -> None:
        """Add a record to the buffer (non-blocking).

        If the buffer reaches ``batch_size``, a flush is scheduled
        immediately.
        """
        if self._closed:
            return
        envelope = TraceEnvelope(type=span_type, data=data)
        self._buffer.append(envelope)
        if len(self._buffer) >= self._config.batch_size:
            self._schedule_flush()

    async def start(self) -> None:
        """Start the background flush loop."""
        self._client = httpx.AsyncClient(
            timeout=self._config.timeout_seconds,
        )
        self._closed = False
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def close(self) -> None:
        """Flush remaining records and tear down resources."""
        self._closed = True
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final drain
        await self._flush()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        """Periodically flush the buffer."""
        try:
            while not self._closed:
                await asyncio.sleep(self._config.flush_interval_seconds)
                await self._flush()
        except asyncio.CancelledError:
            pass

    def _schedule_flush(self) -> None:
        """Schedule an immediate flush without blocking the caller."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._flush())
        except RuntimeError:
            pass  # no running loop — will flush on next timer tick

    async def _flush(self) -> None:
        """Send buffered records to the backend."""
        async with self._lock:
            if not self._buffer:
                return
            batch = self._buffer[:]
            self._buffer.clear()

        # Build NDJSON payload
        lines = [envelope.model_dump_json(exclude_none=True) for envelope in batch]
        payload = "\n".join(lines)

        headers = {
            "Content-Type": "application/x-ndjson",
            **self._config.headers,
        }
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        try:
            if self._client is None:
                return
            resp = await self._client.post(
                self._config.endpoint,
                content=payload,
                headers=headers,
            )
            if resp.status_code >= 400:
                logger.warning(
                    "Rllm ingest returned %s: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except httpx.HTTPError as exc:
            logger.warning("Rllm ingest failed: %s", exc)
        except Exception:
            logger.exception("Unexpected error flushing to Rllm backend.")


# ---------------------------------------------------------------------------
# BigQuery exporter
# ---------------------------------------------------------------------------


class BigQueryValidationError(Exception):
    """Raised when the configured BigQuery dataset or table cannot be found."""


def _epoch_to_iso(epoch: Any) -> str | None:
    """Convert an epoch float (seconds since 1970) to an ISO 8601 UTC string."""
    if epoch is None:
        return None
    try:
        return datetime.datetime.fromtimestamp(float(epoch), tz=datetime.timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _extract_id(span_type: str, data: dict[str, Any]) -> str:
    """Extract the most appropriate ID from a span record.

    Falls back to a random UUID if no known ID field is present.
    """
    if span_type in ("session", "session.start"):
        return data.get("session_id") or str(uuid.uuid4())
    if span_type.startswith("invocation."):
        return data.get("invocation_id") or str(uuid.uuid4())
    if span_type.startswith(("agent.", "llm.", "tool.")):
        return data.get("span_id") or str(uuid.uuid4())
    if span_type == "event":
        return data.get("event_id") or str(uuid.uuid4())
    if span_type.startswith("experiment."):
        if span_type == "experiment.case":
            return data.get("case_id") or str(uuid.uuid4())
        return data.get("experiment_id") or str(uuid.uuid4())
    if span_type.startswith("trajectory."):
        return data.get("trajectory_uid") or data.get("span_id") or str(uuid.uuid4())
    return str(uuid.uuid4())


def get_bq_table_schema() -> list:
    """Return the BigQuery ``SchemaField`` list for the spans table.

    Uses a deferred import so the module remains loadable without
    ``google-cloud-bigquery`` installed.
    """
    from google.cloud.bigquery import SchemaField

    return [
        SchemaField("id", "STRING", mode="REQUIRED", description="Primary ID extracted per span type"),
        SchemaField("span_type", "STRING", mode="REQUIRED", description="Span discriminator (e.g. llm.end, tool.start)"),
        SchemaField("session_id", "STRING", mode="REQUIRED", description="Agent session ID"),
        SchemaField("invocation_id", "STRING", mode="NULLABLE", description="Invocation ID (all except session spans)"),
        SchemaField("span_id", "STRING", mode="NULLABLE", description="Span ID for agent/llm/tool spans"),
        SchemaField("agent_name", "STRING", mode="NULLABLE", description="Name of the agent that produced the span"),
        SchemaField("model", "STRING", mode="NULLABLE", description="LLM model name"),
        SchemaField("tool_name", "STRING", mode="NULLABLE", description="Tool name for tool spans"),
        SchemaField("duration_ms", "FLOAT", mode="NULLABLE", description="Span duration in milliseconds"),
        SchemaField("input_tokens", "INTEGER", mode="NULLABLE", description="LLM input token count"),
        SchemaField("output_tokens", "INTEGER", mode="NULLABLE", description="LLM output token count"),
        SchemaField("error", "STRING", mode="NULLABLE", description="Error message if the span errored"),
        SchemaField("started_at", "TIMESTAMP", mode="NULLABLE", description="Span start time (UTC)"),
        SchemaField("ended_at", "TIMESTAMP", mode="NULLABLE", description="Span end time (UTC)"),
        SchemaField("ingested_at", "TIMESTAMP", mode="REQUIRED", description="Row insertion time (UTC, partition key)"),
        SchemaField("data", "JSON", mode="REQUIRED", description="Full raw span payload as JSON"),
    ]


class BigQueryExporter(BaseExporter):
    """Async batching exporter that writes span records to a Google BigQuery table.

    All span types are written to a single table using a hybrid schema:
    promoted columns for frequently-queried fields, plus a JSON ``data``
    column containing the full raw payload.

    When ``bq_auto_create`` is ``True``, the dataset and table are created
    automatically if they don't exist.  Otherwise :meth:`start` raises
    :class:`BigQueryValidationError` if either is missing.

    The table is partitioned by ``ingested_at`` (DAY) and clustered by
    ``(span_type, session_id)`` for efficient querying.
    """

    def __init__(self, config: RllmConfig) -> None:
        self._config = config
        self._buffer: list[tuple[str, dict[str, Any]]] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._client: Any = None  # bigquery.Client
        self._table_ref: str = ""
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, span_type: SpanType, data: dict[str, Any]) -> None:
        if self._closed:
            return
        self._buffer.append((span_type, data))
        if len(self._buffer) >= self._config.batch_size:
            self._schedule_flush()

    async def start(self) -> None:
        from google.cloud import bigquery

        self._client = bigquery.Client(project=self._config.bq_project)
        dataset_ref = f"{self._config.bq_project}.{self._config.bq_dataset}"
        self._table_ref = f"{dataset_ref}.{self._config.bq_table}"

        loop = asyncio.get_running_loop()

        if self._config.bq_auto_create:
            await self._ensure_dataset_and_table(loop, bigquery, dataset_ref)
        else:
            await self._validate_dataset_and_table(loop, dataset_ref)

        self._closed = False
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("BigQuery exporter started — writing to %s", self._table_ref)

    async def close(self) -> None:
        self._closed = True
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final drain
        await self._flush()
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    async def _validate_dataset_and_table(self, loop: asyncio.AbstractEventLoop, dataset_ref: str) -> None:
        """Raise :class:`BigQueryValidationError` if dataset or table is missing."""
        try:
            await loop.run_in_executor(None, self._client.get_dataset, dataset_ref)
        except Exception as exc:
            raise BigQueryValidationError(f"BigQuery dataset '{self._config.bq_dataset}' not found in project '{self._config.bq_project}'. Set bq_auto_create=True to create it automatically: {exc}") from exc

        try:
            await loop.run_in_executor(None, self._client.get_table, self._table_ref)
        except Exception as exc:
            raise BigQueryValidationError(f"BigQuery table '{self._table_ref}' not found. Set bq_auto_create=True to create it automatically: {exc}") from exc

    async def _ensure_dataset_and_table(self, loop: asyncio.AbstractEventLoop, bigquery: Any, dataset_ref: str) -> None:
        """Create dataset and table if they don't already exist."""
        # --- Dataset ---
        try:
            await loop.run_in_executor(None, self._client.get_dataset, dataset_ref)
            logger.debug("BigQuery dataset '%s' already exists.", dataset_ref)
        except Exception:
            logger.info("Creating BigQuery dataset '%s'...", dataset_ref)
            dataset = bigquery.Dataset(dataset_ref)
            await loop.run_in_executor(
                None,
                lambda: self._client.create_dataset(dataset, exists_ok=True),
            )
            logger.info("BigQuery dataset '%s' created.", dataset_ref)

        # --- Table ---
        try:
            await loop.run_in_executor(None, self._client.get_table, self._table_ref)
            logger.debug("BigQuery table '%s' already exists.", self._table_ref)
        except Exception:
            logger.info("Creating BigQuery table '%s'...", self._table_ref)
            table = bigquery.Table(self._table_ref, schema=get_bq_table_schema())
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="ingested_at",
            )
            table.clustering_fields = ["span_type", "session_id"]
            await loop.run_in_executor(
                None,
                lambda: self._client.create_table(table, exists_ok=True),
            )
            logger.info("BigQuery table '%s' created.", self._table_ref)

    # ------------------------------------------------------------------
    # Flush internals
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(self._config.flush_interval_seconds)
                await self._flush()
        except asyncio.CancelledError:
            pass

    def _schedule_flush(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._flush())
        except RuntimeError:
            pass

    async def _flush(self) -> None:
        async with self._lock:
            if not self._buffer:
                return
            batch = self._buffer[:]
            self._buffer.clear()

        rows = [self._build_row(span_type, data) for span_type, data in batch]

        loop = asyncio.get_running_loop()
        try:
            errors = await loop.run_in_executor(
                None,
                lambda: self._client.insert_rows_json(self._table_ref, rows),
            )
            if errors:
                logger.warning("BigQuery streaming insert errors: %s", errors)
        except Exception as exc:
            from google.api_core.exceptions import NotFound

            if isinstance(exc, NotFound) and self._config.bq_auto_create:
                logger.info("Table not found during flush — auto-creating %s", self._table_ref)
                try:
                    from google.cloud import bigquery

                    dataset_ref = f"{self._config.bq_project}.{self._config.bq_dataset}"
                    await self._ensure_dataset_and_table(loop, bigquery, dataset_ref)
                    # Retry the insert after creation
                    errors = await loop.run_in_executor(
                        None,
                        lambda: self._client.insert_rows_json(self._table_ref, rows),
                    )
                    if errors:
                        logger.warning("BigQuery streaming insert errors (after auto-create): %s", errors)
                except Exception:
                    logger.exception("Failed to auto-create table and retry flush for %d rows.", len(rows))
            else:
                logger.exception("Failed to flush %d rows to BigQuery.", len(rows))

    def _build_row(self, span_type: str, data: dict[str, Any]) -> dict[str, Any]:
        """Transform a span record into a BigQuery row with promoted columns."""
        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

        # Extract promoted fields from nested data
        nested_data = data.get("data", {}) or {}
        request = nested_data.get("request", {}) or {}
        response = nested_data.get("response", {}) or {}
        usage = response.get("usage", {}) or nested_data.get("usage", {}) or {}

        row: dict[str, Any] = {
            "id": _extract_id(span_type, data),
            "span_type": span_type,
            "session_id": data.get("session_id", ""),
            "invocation_id": data.get("invocation_id"),
            "span_id": data.get("span_id"),
            "agent_name": data.get("agent_name"),
            "model": request.get("model") or data.get("model"),
            "tool_name": data.get("tool_name"),
            "duration_ms": data.get("duration_ms"),
            "input_tokens": usage.get("input_tokens") or data.get("input_tokens"),
            "output_tokens": usage.get("output_tokens") or data.get("output_tokens"),
            "error": data.get("error"),
            "started_at": _epoch_to_iso(data.get("started_at")),
            "ended_at": _epoch_to_iso(data.get("ended_at")),
            "ingested_at": now,
            "data": json.dumps(data, default=str),
        }

        # Strip None values — BQ streaming inserts handle missing nullable fields
        return {k: v for k, v in row.items() if v is not None}


# ---------------------------------------------------------------------------
# Agent trajectory exporter (composite)
# ---------------------------------------------------------------------------


class AgentSpanExporter(BaseExporter):
    """Composite exporter that sends ALL span events to the rllm_ui
    backend in real-time (per-record), and also delegates to a wrapped
    inner exporter (e.g. stdout).

    Every span type (``session``, ``invocation.*``, ``agent.*``,
    ``llm.*``, ``tool.*``, ``event``, etc.) is POSTed individually
    as JSON to::

        {agent_endpoint}/api/agent-sessions/{session_id}/spans

    An agent session is automatically created on the backend when
    :meth:`start` is called.
    """

    def __init__(self, config: RllmConfig, inner: BaseExporter) -> None:
        self._config = config
        self._inner = inner
        self._client: httpx.AsyncClient | None = None
        self._closed = False
        self._agent_session_id: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, span_type: SpanType, data: dict[str, Any]) -> None:
        if self._closed:
            return
        if self._agent_session_id:
            self._schedule_send(span_type, data)
        # Always forward to inner (so trajectory spans are also
        # printed to stdout when using the stdout backend)
        self._inner.enqueue(span_type, data)

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=self._config.timeout_seconds)
        self._closed = False
        # Create agent session on the backend
        try:
            self._agent_session_id = await self._create_agent_session()
            logger.info(
                "Agent session created: %s (endpoint: %s)",
                self._agent_session_id,
                self._config.agent_endpoint,
            )
        except Exception as exc:
            logger.exception(
                "Failed to create agent session on %s — trajectory streaming disabled for this run.",
                self._config.agent_endpoint,
            )
            print(
                f"[rllm_telemetry] ERROR: Failed to create agent session on {self._config.agent_endpoint} — trajectory streaming disabled.\n  {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            self._agent_session_id = None
        await self._inner.start()

    async def close(self) -> None:
        self._closed = True
        # Complete the agent session
        if self._agent_session_id and self._client:
            try:
                await self._client.post(
                    f"{self._config.agent_endpoint}/api/agent-sessions/{self._agent_session_id}/complete",
                    headers=self._auth_headers(),
                )
            except Exception as exc:
                logger.warning("Failed to complete agent session %s", self._agent_session_id)
                print(
                    f"[rllm_telemetry] Failed to complete agent session {self._agent_session_id}: {exc}",
                    file=sys.stderr,
                )
        if self._client:
            await self._client.aclose()
            self._client = None
        await self._inner.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        key = self._config.agent_api_key or self._config.api_key
        headers: dict[str, str] = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def _create_agent_session(self) -> str:
        """POST to the backend to register a new agent session."""
        assert self._client is not None
        name = self._config.agent_session_name or f"agent-{uuid.uuid4().hex[:8]}"
        resp = await self._client.post(
            f"{self._config.agent_endpoint}/api/agent-sessions",
            json={"name": name},
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def _schedule_send(self, span_type: SpanType, data: dict[str, Any]) -> None:
        """Fire-and-forget async POST for a single trajectory span."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send_span(span_type, data))
        except RuntimeError:
            pass  # no running loop

    async def _send_span(self, span_type: SpanType, data: dict[str, Any]) -> None:
        """POST a single trajectory span to the backend."""
        if self._client is None or self._agent_session_id is None:
            return
        url = f"{self._config.agent_endpoint}/api/agent-sessions/{self._agent_session_id}/spans"
        payload = {"type": span_type, "data": data}
        try:
            resp = await self._client.post(
                url,
                json=payload,
                headers=self._auth_headers(),
            )
            if resp.status_code >= 400:
                msg = f"[rllm_telemetry] Span ingest returned {resp.status_code}: {resp.text[:200]}"
                logger.warning(msg)
                print(msg, file=sys.stderr)
        except httpx.HTTPError as exc:
            msg = f"[rllm_telemetry] Span ingest failed: {exc}"
            logger.warning(msg)
            print(msg, file=sys.stderr)
        except Exception as exc:
            msg = f"[rllm_telemetry] Unexpected error sending span: {exc}"
            logger.exception(msg)
            print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Registry of built-in exporter backends.
_EXPORTERS: dict[str, type[BaseExporter]] = {
    "http": HttpExporter,
    "stdout": StdoutExporter,
    "bigquery": BigQueryExporter,
}


def create_exporter(config: RllmConfig) -> BaseExporter:
    """Instantiate the exporter specified by ``config.backend``.

    If ``config.agent_endpoint`` is set, the inner exporter is wrapped
    with :class:`AgentSpanExporter` to route all span events to
    the rllm_ui backend.
    """
    cls = _EXPORTERS.get(config.backend)
    if cls is None:
        raise ValueError(f"Unknown exporter backend {config.backend!r}. Choose from: {', '.join(sorted(_EXPORTERS))}")
    inner = cls(config)
    if config.agent_endpoint:
        if not (config.agent_api_key or config.api_key):
            if config.backend == "bigquery":
                # BigQuery uses Google Cloud credentials; skip agent span
                # streaming when no RLLM API key is available.
                return inner
            raise ValueError("An API key is required to stream spans to the rllm-ui backend. Set the RLLM_API_KEY environment variable or pass api_key= / agent_api_key= to RllmConfig.")
        return AgentSpanExporter(config, inner)
    return inner


# Backward-compat aliases
RllmExporter = HttpExporter
AgentTrajectoryExporter = AgentSpanExporter

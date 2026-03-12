"""Exporters that ship trace records to various backends.

The :class:`BaseExporter` protocol defines the interface.  Three concrete
implementations are provided:

* :class:`HttpExporter` — async batching HTTP exporter (NDJSON).
* :class:`StdoutExporter` — pretty-prints traces to stdout for local testing.
* :class:`AgentSpanExporter` — composite exporter that routes all span
  events to the rllm_ui backend in real-time (per-record).
"""

from __future__ import annotations

import asyncio
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
    "invocation.start": "\033[34m",  # blue
    "invocation.end": "\033[34m",
    "agent.start": "\033[35m",  # magenta
    "agent.end": "\033[35m",
    "llm.start": "\033[33m",  # yellow
    "llm.end": "\033[33m",
    "tool.start": "\033[32m",  # green
    "tool.end": "\033[32m",
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
        except Exception:
            logger.exception(
                "Failed to create agent session on %s — trajectory streaming disabled for this run.",
                self._config.agent_endpoint,
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
            except Exception:
                logger.warning("Failed to complete agent session %s", self._agent_session_id)
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
                logger.warning(
                    "Agent trajectory ingest returned %s: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except httpx.HTTPError as exc:
            logger.warning("Agent trajectory ingest failed: %s", exc)
        except Exception:
            logger.exception("Unexpected error sending trajectory span.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Registry of built-in exporter backends.
_EXPORTERS: dict[str, type[BaseExporter]] = {
    "http": HttpExporter,
    "stdout": StdoutExporter,
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
        return AgentSpanExporter(config, inner)
    return inner


# Backward-compat aliases
RllmExporter = HttpExporter
AgentTrajectoryExporter = AgentSpanExporter

"""Exporters that ship trace records to various backends.

The :class:`BaseExporter` protocol defines the interface.  Two concrete
implementations are provided:

* :class:`HttpExporter` — async batching HTTP exporter (NDJSON).
* :class:`StdoutExporter` — pretty-prints traces to stdout for local testing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
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
# Factory
# ---------------------------------------------------------------------------

#: Registry of built-in exporter backends.
_EXPORTERS: dict[str, type[BaseExporter]] = {
    "http": HttpExporter,
    "stdout": StdoutExporter,
}


def create_exporter(config: RllmConfig) -> BaseExporter:
    """Instantiate the exporter specified by ``config.backend``."""
    cls = _EXPORTERS.get(config.backend)
    if cls is None:
        raise ValueError(f"Unknown exporter backend {config.backend!r}. Choose from: {', '.join(sorted(_EXPORTERS))}")
    return cls(config)


# Backward-compat alias
RllmExporter = HttpExporter

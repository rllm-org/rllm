"""Base protocol and utilities for tracer implementations."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class TracerProtocol(Protocol):
    """
    Common interface for all tracer implementations.

    All tracer types (in-memory, persistent, etc.) must implement this protocol.
    This allows chat clients to work with any tracer through a uniform API.

    Example:
        >>> from rllm.sdk.tracers import InMemorySessionTracer, SqliteTracer
        >>>
        >>> # Use in-memory tracer
        >>> tracer = InMemorySessionTracer()
        >>> llm = get_chat_client(tracer=tracer, ...)
        >>>
        >>> # Or use persistent tracer
        >>> tracer = SqliteTracer(db_path="traces.db")
        >>> llm = get_chat_client(tracer=tracer, ...)
        >>>
        >>> # Chat client calls tracer.log_llm_call() - works with any tracer
    """

    def log_llm_call(
        self,
        name: str,
        input: str | list | dict,
        output: str | dict,
        model: str,
        latency_ms: float,
        tokens: dict[str, int],
        session_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
        parent_trace_id: str | None = None,
        cost: float | None = None,
        environment: str | None = None,
        tools: list[dict] | None = None,
        contexts: list[str | dict] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """
        Log an LLM call trace.

        Args:
            name: Identifier for the call (e.g., "chat.completions.create")
            input: Input data (messages, prompt, etc.)
            output: Output data (response, completion, etc.)
            model: Model identifier (e.g., "gpt-4")
            latency_ms: Latency in milliseconds
            tokens: Token usage dict with keys: prompt, completion, total
            session_name: Session name (optional, may be inferred from context)
            metadata: Additional metadata dict
            trace_id: Unique trace ID (auto-generated if None)
            parent_trace_id: Parent trace ID for nested calls
            cost: Cost in USD (optional)
            environment: Environment name (e.g., "production", "dev")
            tools: List of tool definitions used
            contexts: List of context IDs or dicts
            tags: List of tags for categorization
        """
        ...

    def flush(self, timeout: float = 30.0) -> bool | None:
        """
        Flush all pending traces (blocking).

        This method should block until all traces are persisted or
        the timeout is reached.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if flush succeeded, False if it failed/timed out,
            or None for backward compatibility (treated as success)
        """
        ...

    async def close(self, timeout: float = 30.0) -> None:
        """
        Close tracer and flush pending traces.

        This method should clean up resources and ensure all traces
        are persisted before returning.

        Args:
            timeout: Maximum time to wait in seconds
        """
        ...

    def __repr__(self):
        return f"CompositeTracer(tracers={[t.__class__.__name__ for t in self.tracers]})"

"""In-memory session tracer for immediate access to traces."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from rllm.sdk.protocol import Trace
from rllm.sdk.session import get_active_sessions


class InMemorySessionTracer:
    """
    In-memory tracer that appends traces to all active sessions.

    This tracer enables immediate access to LLM call traces via `session.llm_calls`
    with zero I/O overhead. It automatically detects the active session stack from
    context and appends formatted traces to each session's in-memory call list
    (outer and inner).

    Features:
    - Zero I/O - all data stays in memory
    - Automatic session stack detection from context
    - Optional trace formatting
    - Immediate access via session.llm_calls
    - Works with nested sessions (adds to both outer and inner)

    Design:
    1. On log_llm_call(), reads get_active_sessions()
    2. If sessions exist, formats trace and appends to each session._calls
    3. If no session, trace is dropped (no global buffering)

    This tracer does NOT persist anything - it only populates the
    in-memory session._calls list(s) for immediate access.

    Example:
        >>> from rllm.sdk import SessionContext, get_chat_client
        >>> from rllm.sdk.tracers import InMemorySessionTracer
        >>>
        >>> # Create tracer
        >>> tracer = InMemorySessionTracer()
        >>>
        >>> # Create chat client with tracer
        >>> llm = get_chat_client(tracer=tracer, model="gpt-4")
        >>>
        >>> # Use within session
        >>> with SessionContext() as session:
        ...     llm.chat.completions.create(
        ...         messages=[{"role": "user", "content": "Hello"}]
        ...     )
        ...
        ...     # Immediate access - zero I/O!
        ...     print(f"Calls: {len(session.llm_calls)}")
        ...     print(session.llm_calls[0]["model"])  # "gpt-4"

        With custom formatter:
        >>> def my_formatter(trace: dict) -> dict:
        ...     # Only keep essential fields
        ...     return {
        ...         "model": trace["model"],
        ...         "prompt": trace["input"]["messages"],
        ...         "response": trace["output"],
        ...         "tokens": trace["tokens"],
        ...     }
        >>>
        >>> tracer = InMemorySessionTracer(formatter=my_formatter)
        >>> # Traces are formatted before appending to session
    """

    def __init__(self, formatter: Callable[[dict], dict] | None = None):
        """
        Initialize in-memory session tracer.

        Args:
            formatter: Optional function to transform trace data before
                      appending to session. Receives raw trace dict,
                      returns formatted trace dict.
                      Default: identity function (no transformation)
        """
        self.formatter = formatter or (lambda x: x)

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
        Log trace to all active sessions' in-memory storage.

        Algorithm:
        1. Get active sessions from context via get_active_sessions()
        2. If none, return early (trace is dropped)
        3. Build trace dict with all provided data
        4. Apply formatter (if configured)
        5. Append formatted trace to each session._calls

        Args:
            name: Identifier for the call (e.g., "chat.completions.create")
            input: Input data (messages, prompt, etc.)
            output: Output data (response, completion, etc.)
            model: Model identifier (e.g., "gpt-4")
            latency_ms: Latency in milliseconds
            tokens: Token usage dict with keys: prompt, completion, total
            session_name: Session name (ignored - session names come from active sessions in context)
            metadata: Additional metadata dict
            trace_id: Unique trace ID (auto-generated if None, or extracted from output.id)
            parent_trace_id: Parent trace ID for nested calls
            cost: Cost in USD (optional)
            environment: Environment name (e.g., "production", "dev")
            tools: List of tool definitions used
            contexts: List of context IDs or dicts
            tags: List of tags for categorization

        Note:
            - The `session_name` parameter is ignored. Session names are automatically
              extracted from active sessions in context via `get_active_sessions()`.
              Each active session gets its own trace with its own session_name.
            - If not within a session context (no active sessions found),
              the trace is silently dropped. This is intentional - in-memory tracer
              only works within sessions.
        """
        # Get all active sessions (outer → inner)
        sessions = get_active_sessions()

        if not sessions:
            # Not in a session context - nothing to do
            # In-memory tracer only works within sessions
            return

        # Extract trace_id: prefer provided trace_id, then check output for id, otherwise generate
        if trace_id is None:
            # Check if output contains an id field (common in LLM provider responses)
            if isinstance(output, dict):
                trace_id = output.get("id")

        # Use single trace_id for the same logical call across sessions
        actual_trace_id = trace_id or f"tr_{uuid.uuid4().hex[:16]}"

        # Build base trace data (session_name will be set per-session)
        trace_kwargs = {
            "trace_id": actual_trace_id,
            "name": name,
            "input": input,
            "output": output,
            "model": model,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "parent_trace_id": parent_trace_id,
            "cost": cost,
            "environment": environment,
            "tools": tools,
            "contexts": contexts,
            "tags": tags,
        }

        # Add trace to every active session's storage with its own session_name
        for sess in sessions:
            trace_obj = Trace(session_name=sess.name, **trace_kwargs)
            # Add to session storage with full UID chain for tree hierarchy
            sess.storage.add_trace(sess._session_uid_chain, sess.name, trace_obj)

    def flush(self, timeout: float = 30.0) -> bool:
        """
        No-op for in-memory tracer.

        In-memory tracer has no buffering or background workers,
        so there's nothing to flush.

        Args:
            timeout: Ignored (kept for protocol compatibility)

        Returns:
            True (always succeeds since it's a no-op)
        """
        return True

    async def close(self, timeout: float = 30.0) -> None:
        """
        No-op for in-memory tracer.

        In-memory tracer has no resources to clean up.

        Args:
            timeout: Ignored (kept for protocol compatibility)
        """
        pass

    def __repr__(self):
        return f"InMemorySessionTracer(formatter={self.formatter.__name__ if hasattr(self.formatter, '__name__') else 'custom'})"

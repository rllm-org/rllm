"""In-memory session tracer for immediate access to traces."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rllm.sdk.data_process import build_llm_io
from rllm.sdk.protocol import Trace

if TYPE_CHECKING:
    from rllm.sdk.session.contextvar import ContextVarSession


class InMemorySessionTracer:
    """
    In-memory tracer that appends traces to provided sessions.

    This tracer enables immediate access to LLM call traces via `session.llm_calls`
    with zero I/O overhead. Callers must provide the session objects to store to.

    Features:
    - Zero I/O - all data stays in memory
    - No auto-detection - caller provides sessions explicitly
    - Optional trace formatting
    - Immediate access via session.llm_calls
    - Works with nested sessions (adds to all provided sessions)

    Design:
    1. On log_llm_call(), uses provided sessions parameter
    2. If sessions provided, formats trace and appends to each session's storage
    3. If no sessions provided, trace is dropped

    This tracer does NOT persist anything - it only populates the
    in-memory session storage for immediate access.

    Example:
        >>> from rllm.sdk import SessionContext, get_chat_client
        >>> from rllm.sdk.tracers import InMemorySessionTracer
        >>> from rllm.sdk.session import get_active_cv_sessions
        >>>
        >>> tracer = InMemorySessionTracer()
        >>> llm = get_chat_client(tracer=tracer)
        >>>
        >>> with SessionContext() as session:
        ...     llm.chat.completions.create(model="gpt-4", messages=[...])
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
        session_uids: list[str] | None = None,
        sessions: list[ContextVarSession] | None = None,
    ) -> None:
        """
        Log trace to provided sessions' in-memory storage.

        This method does NOT perform any autofill or context lookups.
        All values are used as-is from the caller. If trace_id is None,
        a new one is generated. All other None values remain None.

        IMPORTANT: Caller MUST provide sessions explicitly via the `sessions`
        parameter. This tracer does not auto-detect from context.

        Algorithm:
        1. Use provided sessions parameter
        2. If no sessions provided, return early (trace is dropped)
        3. Build trace dict with all provided data (no overwrites)
        4. Append trace to each session's storage

        Args:
            name: Identifier for the call (e.g., "chat.completions.create")
            input: Input data (messages, prompt, etc.)
            output: Output data (response, completion, etc.)
            model: Model identifier (e.g., "gpt-4")
            latency_ms: Latency in milliseconds
            tokens: Token usage dict with keys: prompt, completion, total
            session_name: Session name (caller must provide, no auto-detection)
            metadata: Additional metadata dict (caller must provide, no merging)
            trace_id: Unique trace ID (caller should provide, auto-generated if None)
            parent_trace_id: Parent trace ID for nested calls
            cost: Cost in USD (optional)
            environment: Environment name (e.g., "production", "dev")
            tools: List of tool definitions used
            contexts: List of context IDs or dicts
            tags: List of tags for categorization
            session_uids: List of session UIDs (for protocol compatibility)
            sessions: List of session objects to store traces to (REQUIRED for this tracer)

        Note:
            - If sessions is None or empty, trace is silently dropped.
            - Caller must provide sessions via get_active_cv_sessions() or similar.
        """
        if not sessions:
            # No sessions provided - nothing to do
            return

        # Generate trace_id only if not provided (no extraction from output)
        actual_trace_id = trace_id or f"tr_{uuid.uuid4().hex[:16]}"

        # Build trace data - use all values as-is (no overwrites)
        llm_input, llm_output = build_llm_io(input, output)

        trace_kwargs = {
            "trace_id": actual_trace_id,
            "session_name": session_name or "",
            "name": name,
            "input": llm_input,
            "output": llm_output,
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

        # Add trace to every active session's storage
        for sess in sessions:
            trace_obj = Trace(**trace_kwargs)
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

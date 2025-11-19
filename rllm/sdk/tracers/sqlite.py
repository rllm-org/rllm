"""SQLite-based persistent tracer using SqliteTraceStore."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import uuid
from typing import Any

from rllm.sdk.protocol import Trace
from rllm.sdk.session import get_active_sessions, get_current_metadata, get_current_session_name
from rllm.sdk.store import SqliteTraceStore

logger = logging.getLogger(__name__)


class SqliteTracer:
    """
    Persistent tracer backed by SQLite database.

    This tracer provides durable storage for LLM traces using a local SQLite database.
    Traces are queued and stored asynchronously by a background worker that awaits
    each store operation to completion, ensuring durability while keeping logging non-blocking.

    Features:
    - Fast queries by session UID (indexed junction table)
    - Automatic session context extraction
    - Non-blocking logging (returns immediately)
    - Background worker awaits store operations to completion
    - Retry logic with exponential backoff (3 attempts: 1s, 2s, 4s delays)
    - Simple, standalone implementation (no external dependencies)

    Example:
        >>> from rllm.sdk.tracers import SqliteTracer
        >>> from rllm.sdk import get_chat_client
        >>>
        >>> tracer = SqliteTracer(db_path="traces.db")
        >>> llm = get_chat_client(tracer=tracer, model="gpt-4")
        >>>
        >>> with SessionContext() as session:
        ...     llm.chat.completions.create(...)
        ... # Trace is stored in SQLite with session UID
        >>>
        >>> # Query traces by session UID
        >>> traces = await tracer.store.get_by_session_uid(session._uid)
    """

    # Sentinel used to unblock the worker on shutdown
    _STOP = object()

    def __init__(
        self,
        db_path: str | None = None,
        namespace: str = "default",
        max_queue_size: int = 10000,
    ):
        """
        Initialize SQLite tracer.

        Args:
            db_path: Path to SQLite database file. If None, uses default location
                    (~/.rllm/traces.db or temp directory)
            namespace: Namespace for organizing traces (default: "default")
            max_queue_size: Maximum number of traces to buffer in memory (default: 10,000)
        """
        self.store = SqliteTraceStore(db_path=db_path)
        self.namespace = namespace

        # Queue for storing traces (thread-safe)
        self._store_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue_size)
        self._max_queue_size = max_queue_size
        self._shutdown = False

        # Background worker thread and event loop
        self._worker_thread: threading.Thread | None = None
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._worker_started = threading.Event()

        # Start background worker
        self._start_background_worker()

    def _start_background_worker(self) -> None:
        """Start the background worker thread with its own event loop."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._worker_thread = threading.Thread(target=self._run_worker_loop, daemon=True, name="SqliteTracer-Worker")
        self._worker_thread.start()

        # Wait for worker to start
        self._worker_started.wait(timeout=5.0)

    def _run_worker_loop(self) -> None:
        """Run the worker event loop forever in a separate thread."""
        # Create a new event loop for this thread
        self._worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._worker_loop)

        # Signal that worker is ready
        self._worker_started.set()

        # Schedule the async worker as a task
        self._worker_loop.create_task(self._worker_coroutine())

        # Run the loop forever until explicitly stopped
        try:
            self._worker_loop.run_forever()
        finally:
            # Close the loop only after it has been stopped
            if self._worker_loop and not self._worker_loop.is_closed():
                self._worker_loop.close()
            self._worker_loop = None

    async def _worker_coroutine(self) -> None:
        """Main worker coroutine that processes the store queue."""
        while True:
            # Blocking get for the next item
            item = self._store_queue.get()

            # Check for stop sentinel
            if item is self._STOP:
                logger.info("[SqliteTracer._worker_coroutine] Stop sentinel received, exiting worker")
                break

            trace_id = item.get("trace_id", "unknown")

            try:
                # Store the trace with retry logic
                await self._store_trace_with_retry(item)
                # logger.info(f"[SqliteTracer._worker_coroutine] Successfully stored trace_id={trace_id}")
            except Exception as e:
                logger.exception(f"[SqliteTracer._worker_coroutine] Worker error processing trace {trace_id}: {e}")
            finally:
                # Mark task as done
                self._store_queue.task_done()

        if self._worker_loop and not self._worker_loop.is_closed():
            self._worker_loop.call_soon(self._worker_loop.stop)

    async def _store_trace_with_retry(self, item: dict[str, Any]) -> None:
        """Store a trace with retry logic."""
        max_retries = 3
        retry_delays = [1, 2, 4]  # seconds
        trace_id = item.get("trace_id", "unknown")
        session_uids = item.get("session_uids")

        for attempt in range(max_retries):
            try:
                await self.store.store(
                    trace_id=item["trace_id"],
                    data=item["data"],
                    namespace=item["namespace"],
                    context_type=item["context_type"],
                    metadata=item.get("metadata"),
                    session_uids=session_uids,
                )
                return  # Success, exit
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"[SqliteTracer._store_trace_with_retry] Failed to store trace {trace_id} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delays[attempt]}s...")
                    await asyncio.sleep(retry_delays[attempt])
                else:
                    logger.exception(f"[SqliteTracer._store_trace_with_retry] Dropping trace {trace_id} after {max_retries} failed attempts: {e}")

    def _stop_worker_loop(self) -> None:
        """Stop the worker event loop gracefully."""
        if self._worker_loop is None:
            return

        try:

            def _stop():
                if self._worker_loop and not self._worker_loop.is_closed():
                    self._worker_loop.stop()

            self._worker_loop.call_soon_threadsafe(_stop)
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5.0)
            if self._worker_loop and not self._worker_loop.is_closed():
                self._worker_loop.close()
        except Exception as e:
            logger.exception("Error stopping worker loop: %s", e)
        finally:
            self._worker_loop = None

    def _queue_trace(
        self,
        trace: Trace,
        metadata: dict[str, Any] | None = None,
        session_uids: list[str] | None = None,
    ) -> None:
        """Queue trace for storage (non-blocking, will be awaited by worker)."""
        # Ensure background worker is running
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._start_background_worker()

        try:
            # Convert Trace object to dict for storage
            trace_data = trace.model_dump()

            queue_item = {
                "trace_id": trace.trace_id,
                "data": trace_data,
                "namespace": self.namespace,
                "context_type": "llm_trace",
                "metadata": metadata,
                "session_uids": session_uids,
            }
            self._store_queue.put_nowait(queue_item)
        except queue.Full:
            logger.warning(f"Store queue full (max size: {self._max_queue_size}), dropping trace {trace.trace_id}")

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
    ) -> None:
        """
        Log an LLM call to SQLite store (non-blocking).

        This method returns immediately after queuing the trace. The trace is stored
        asynchronously by a background worker that awaits the store operation to completion.

        Args:
            name: Identifier for the call (e.g., "chat.completions.create")
            input: Input data (messages, prompt, etc.)
            output: Output data (response, completion, etc.)
            model: Model identifier (e.g., "gpt-4")
            latency_ms: Latency in milliseconds
            tokens: Token usage dict with keys: prompt, completion, total
            session_name: Session name (optional, extracted from context if available)
            metadata: Additional metadata dict
            trace_id: Unique trace ID (auto-generated if None, or extracted from output.id)
            parent_trace_id: Parent trace ID for nested calls
            cost: Cost in USD (optional)
            environment: Environment name (e.g., "production", "dev")
            tools: List of tool definitions used
            contexts: List of context IDs or dicts
            tags: List of tags for categorization
            session_uids: List of session UIDs to associate with this trace (optional, auto-detected from context if not provided)
        """
        # Extract trace_id: prefer provided trace_id, then check output for id, otherwise generate
        if trace_id is None:
            # Check if output contains an id field (common in LLM provider responses)
            if isinstance(output, dict):
                trace_id = output.get("id")

        # Generate trace ID if still not available
        if trace_id is None:
            trace_id = f"tr_{uuid.uuid4().hex[:16]}"

        # Get session_name from context if not provided
        if session_name is None:
            session_name = get_current_session_name()

        # Merge context metadata with call-specific metadata
        context_meta = get_current_metadata()
        final_metadata = {**context_meta, **(metadata or {})}

        # Get session UIDs: use provided session_uids, or auto-detect from active sessions
        if session_uids is None:
            active_sessions = get_active_sessions()
            session_uids = [s._uid for s in active_sessions]

        # Create Trace object following the protocol
        trace = Trace(
            trace_id=trace_id,
            session_name=session_name or "",  # Trace protocol requires session_name
            name=name,
            input=input,
            output=output,
            model=model,
            latency_ms=latency_ms,
            tokens=tokens,
            metadata=final_metadata,
            timestamp=time.time(),
            parent_trace_id=parent_trace_id,
            cost=cost,
            environment=environment,
            tools=tools,
            contexts=contexts,
            tags=tags,
        )

        # Queue trace for storage (non-blocking, worker will await it)
        self._queue_trace(
            trace=trace,
            metadata=final_metadata,
            session_uids=session_uids if session_uids else None,
        )

    def flush(self, timeout: float = 30.0) -> bool:
        """
        Block until all queued traces are persisted (synchronous version).

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all traces were flushed successfully, False otherwise
        """
        try:
            # Create a new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the async flush - wait for queue to be processed
            loop.run_until_complete(asyncio.wait_for(asyncio.to_thread(self._store_queue.join), timeout=timeout))
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Flush timeout after {timeout}s, queue still has {self._store_queue.qsize()} items")
            return False
        except Exception as e:
            logger.exception(f"Flush failed with error: {e}")
            return False

    async def store_signal(
        self,
        context_id: str,
        context_type: str = "trace_batch_end",
        data: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Store a signal (temporarily no-op, will be implemented later)."""
        # TODO: Implement signal storage
        pass

    async def close(self, timeout: float = 30.0) -> None:
        """
        Stop the background worker and close the tracer.

        Args:
            timeout: Maximum time to wait in seconds
        """
        self._shutdown = True
        # Unblock the worker if it's waiting for new items
        try:
            self._store_queue.put_nowait(self._STOP)
        except Exception:
            pass
        self._stop_worker_loop()

    def close_sync(self, timeout: float = 30.0) -> None:
        """
        Synchronous close without waiting.

        Args:
            timeout: Maximum time to wait in seconds
        """
        self._shutdown = True
        try:
            self._store_queue.put_nowait(self._STOP)
        except Exception:
            pass
        self._stop_worker_loop()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False

    def __repr__(self):
        return f"SqliteTracer(namespace={self.namespace!r}, db_path={self.store.db_path!r})"

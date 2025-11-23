"""Convenient shortcuts for common SDK operations.

This module provides standalone functions for common SDK operations,
making the SDK more ergonomic for simple use cases.
"""

from __future__ import annotations

from typing import Any

from rllm.sdk.chat import (
    OpenTelemetryTrackedAsyncChatClient,
    OpenTelemetryTrackedChatClient,
    ProxyTrackedAsyncChatClient,
    ProxyTrackedChatClient,
)
from rllm.sdk.session import SESSION_BACKEND, SessionContext, otel_session


def _session_with_name(name: str | None = None, **metadata: Any):
    """Create a session context manager with explicit name (INTERNAL USE ONLY).

    This is an internal function that allows setting an explicit session name.
    For public use, use the `session()` function which auto-generates the name.

    Args:
        name: Explicit session name (auto-generated if None)
        **metadata: Arbitrary metadata to attach to all traces in this session

    Returns:
        SessionContext: A context manager that sets session name and metadata
    """
    if SESSION_BACKEND == "opentelemetry":
        if otel_session is None:
            raise RuntimeError("OpenTelemetry backend requested but opentelemetry package not installed")
        return otel_session(name=name, **metadata)
    return SessionContext(name=name, **metadata)


def session(**metadata: Any):
    """Create session context for automatic trace tracking with auto-generated name.

    Session name is auto-generated. Nested sessions inherit parent metadata.
    For internal use with explicit names, use _session_with_name() instead.

    Args:
        **metadata: Metadata attached to all traces in this session.

    Returns:
        SessionContext: Context manager for session and metadata.

    Example:
        >>> with session(experiment="v1"):
        ...     llm.chat.completions.create(...)  # Traces get metadata
    """
    assert "name" not in metadata, "name is auto-generated and cannot be specified"
    if SESSION_BACKEND == "opentelemetry":
        if otel_session is None:
            raise RuntimeError("OpenTelemetry backend requested but opentelemetry package not installed")
        return otel_session(**metadata)
    return SessionContext(**metadata)


def get_chat_client(
    provider: str = "openai",
    *,
    use_proxy: bool = True,
    **kwargs: Any,
):
    """Get OpenAI chat client with automatic session tracking.

    Returns TrackedChatClient that tracks calls within session() contexts.
    All OpenAI client arguments (api_key, base_url, etc.) are passed through.

    Args:
        provider: Provider name (only "openai" supported).
        use_proxy: Enable proxy features (default: True).
        **kwargs: Passed directly to OpenAI client (api_key, base_url, etc.)

    Returns:
        TrackedChatClient: OpenAI client with session tracking.

    Example:
        >>> llm = get_chat_client(api_key="sk-...")
        >>> with session(experiment="v1"):
        ...     llm.chat.completions.create(model="gpt-4", messages=[...])
    """
    if provider.lower() != "openai":
        raise ValueError(f"Unsupported chat provider '{provider}'. Only 'openai' is supported.")

    # Select client based on session backend
    # OTel backend disables local tracing (handled by OTel)
    # ContextVar backend enables local tracing
    if SESSION_BACKEND == "opentelemetry":
        return OpenTelemetryTrackedChatClient(use_proxy=use_proxy, **kwargs)
    else:
        return ProxyTrackedChatClient(use_proxy=use_proxy, **kwargs)


def get_chat_client_async(
    provider: str = "openai",
    *,
    use_proxy: bool = True,
    **kwargs: Any,
):
    """Get async OpenAI chat client with automatic session tracking.

    Async version of get_chat_client(). See get_chat_client() for details.

    Returns:
        TrackedAsyncChatClient: Async OpenAI client with session tracking.

    Example:
        >>> llm = get_chat_client_async(api_key="sk-...")
        >>> with session(experiment="v1"):
        ...     await llm.chat.completions.create(model="gpt-4", messages=[...])
    """
    if provider.lower() != "openai":
        raise ValueError(f"Unsupported chat provider '{provider}'. Only 'openai' is supported.")

    if SESSION_BACKEND == "opentelemetry":
        return OpenTelemetryTrackedAsyncChatClient(use_proxy=use_proxy, **kwargs)
    else:
        return ProxyTrackedAsyncChatClient(use_proxy=use_proxy, **kwargs)

"""Chat provider clients exposed by the RLLM SDK.

Architecture:
    TrackedChatClient (core)
        ├── use_proxy: bool = True      (inject metadata into proxy URL)
        ├── enable_local_tracing: bool = True  (log to local tracer)
        └── tracer: Any = None          (custom tracer, default: shared in-memory)

    Aliases (preset configurations):
        ProxyTrackedChatClient         = TrackedChatClient (defaults)
        OpenTelemetryTrackedChatClient = TrackedChatClient(enable_local_tracing=False)
"""

from rllm.sdk.chat.openai import (
    # Backward-compatible aliases
    AsyncOpenAIOTelClient,
    OpenAIOTelClient,
    OpenTelemetryTrackedAsyncChatClient,
    OpenTelemetryTrackedChatClient,
    ProxyTrackedAsyncChatClient,
    ProxyTrackedChatClient,
    # Core clients (new)
    TrackedAsyncChatClient,
    TrackedChatClient,
)
from rllm.sdk.chat.util import extract_completion_tokens, extract_usage_tokens, merge_args

__all__ = [
    # Utilities
    "merge_args",
    "extract_completion_tokens",
    "extract_usage_tokens",
    # Core clients
    "TrackedChatClient",
    "TrackedAsyncChatClient",
    # Aliases
    "ProxyTrackedChatClient",
    "ProxyTrackedAsyncChatClient",
    "OpenTelemetryTrackedChatClient",
    "OpenTelemetryTrackedAsyncChatClient",
    "OpenAIOTelClient",
    "AsyncOpenAIOTelClient",
]

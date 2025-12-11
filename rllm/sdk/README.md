# RLLM SDK - Automatic LLM Trace Collection and RL Training

Lightweight SDK for automatic LLM trace collection using session contexts and trajectory decorators. Supports both in-process context tracking (ContextVar) and distributed tracing (OpenTelemetry with W3C baggage).

## Installation

The SDK is part of the `rllm` package:

```python
from rllm.sdk import session, get_chat_client, trajectory
```

For OpenTelemetry support, install with extras:

```bash
pip install rllm[otel]
```

## Quick Start

### Basic Session Usage

```python
from rllm.sdk import session, get_chat_client

llm = get_chat_client(api_key="sk-...")

# Create a session to track all LLM calls
with session(experiment="v1") as sess:
    response = llm.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
    # Access all traces from this session
    print(f"Collected {len(sess.llm_calls)} traces")
```

### Trajectory Decorator

```python
from rllm.sdk import trajectory, get_chat_client_async

llm = get_chat_client_async(api_key="sk-...")

@trajectory(name="solver")
async def solve_math_problem(problem: str):
    # Each LLM call automatically becomes a step
    response1 = await llm.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Solve: {problem}"}]
    )
    response2 = await llm.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Is this correct?"}]
    )
    return response2.choices[0].message.content

# Returns TrajectoryView instead of string
traj = await solve_math_problem("What is 2+2?")
print(f"Steps: {len(traj.steps)}")  # 2
traj.steps[0].reward = 1.0  # Set rewards on each step
traj.reward = sum(s.reward for s in traj.steps)
```

## Core Concepts

### 1. Session Backend Configuration

The SDK supports two session backends, configured via `rllm/sdk/config.yaml`:

```yaml
# config.yaml
session_backend: "contextvar"  # or "opentelemetry"
```

- **contextvar** (default): In-process context tracking using Python's `contextvars`
- **opentelemetry**: Distributed tracing with W3C baggage propagation for cross-process context

### 2. Session Context

Tracks all LLM calls within a context for debugging and analysis.

```python
from rllm.sdk import session

# Auto-generated session name
with session(experiment="v1") as sess:
    llm.chat.completions.create(...)
    print(sess.llm_calls)  # List of Trace objects
```

### 3. Metadata Inheritance

Nested sessions automatically merge metadata.

```python
with session(experiment="v1"):
    with session(task="math"):
        # All traces get: {experiment: "v1", task: "math"}
        llm.chat.completions.create(...)
```

### 4. OpenTelemetry Sessions (Distributed Tracing)

For cross-process context propagation, use the OpenTelemetry backend:

```python
from rllm.sdk.session import otel_session, configure_default_tracer

# Configure tracer once per process
configure_default_tracer(service_name="my-agent")

# Sessions use W3C baggage for context propagation
with otel_session(name="client") as client_session:
    # HTTP calls automatically carry session context via baggage headers
    httpx.post("http://server/api", ...)

# On the server side:
with otel_session(name="handler") as server_session:
    llm.chat.completions.create(...)
    # server_session automatically inherits client's UID chain
```

Key features of OpenTelemetry sessions:
- W3C baggage as single source of truth for session state
- Automatic context propagation across HTTP boundaries
- Span-based session UIDs for distributed tracing
- Compatible with OpenTelemetry observability tools

### 5. Storage Backends

The SDK uses in-memory storage by default for session traces.

```python
from rllm.sdk import session

# In-memory (default)
with session() as sess:
    llm.call()
```

## Proxy Integration

The SDK includes a proxy module for routing LLM calls through a LiteLLM proxy with metadata injection.

### Metadata Routing Middleware

```python
from rllm.sdk.proxy import MetadataRoutingMiddleware

# Add to your ASGI app
app = MetadataRoutingMiddleware(app)
```

### LiteLLM Callbacks

```python
from rllm.sdk.proxy import TracingCallback, SamplingParametersCallback

# Use with LiteLLM for automatic trace collection
callbacks = [TracingCallback(), SamplingParametersCallback()]
```

### Metadata Slug Encoding

```python
from rllm.sdk.proxy import encode_metadata_slug, decode_metadata_slug, build_proxied_base_url

# Encode metadata into URL path for proxy routing
metadata = {"session_name": "my-session", "experiment": "v1"}
slug = encode_metadata_slug(metadata)
proxied_url = build_proxied_base_url("http://localhost:8000", metadata)
```

## API Reference

### Core Functions

```python
# Session management
session(**metadata) -> SessionContext  # Auto-generates session name
get_current_session() -> ContextVarSession | None  # ContextVar backend only
get_current_session_name() -> str | None
get_current_metadata() -> dict
get_active_session_uids() -> list[str]  # Get UID chain for current session

# OpenTelemetry sessions (when backend="opentelemetry")
otel_session(name=None, **metadata) -> OpenTelemetrySession
configure_default_tracer(service_name="rllm-worker") -> None

# Chat clients
get_chat_client(api_key, base_url, ...) -> ProxyTrackedChatClient | OpenTelemetryTrackedChatClient
get_chat_client_async(api_key, base_url, ...) -> ProxyTrackedAsyncChatClient | OpenTelemetryTrackedAsyncChatClient

# Decorators
@trajectory(name: str, **metadata) -> Callable
```

### Session Configuration

```python
from rllm.sdk.session import SESSION_BACKEND

# Check current backend
print(SESSION_BACKEND)  # "contextvar" or "opentelemetry"
```

### Chat Clients

```python
# ContextVar backend (default)
from rllm.sdk.chat import ProxyTrackedChatClient, ProxyTrackedAsyncChatClient

# OpenTelemetry backend
from rllm.sdk.chat import OpenTelemetryTrackedChatClient, OpenTelemetryTrackedAsyncChatClient
# Aliases
from rllm.sdk.chat import OpenAIOTelClient, AsyncOpenAIOTelClient
```

### Proxy Module

```python
from rllm.sdk.proxy import (
    # Middleware
    MetadataRoutingMiddleware,
    # LiteLLM callbacks
    TracingCallback,
    SamplingParametersCallback,
    # Metadata encoding
    encode_metadata_slug,
    decode_metadata_slug,
    build_proxied_base_url,
    extract_metadata_from_path,
    assemble_routing_metadata,
)
```

### Data Models

```python
# Low-level trace from a single LLM call
class Trace:
    trace_id: str
    session_name: str
    input: str | list | dict
    output: str | dict
    model: str
    tokens: dict
    ...

# Trace with reward field (auto-generated from traces)
class StepView:
    id: str
    input: str | list | dict
    output: str | dict
    reward: float
    ...

# Collection of steps forming a trajectory
class TrajectoryView:
    name: str
    steps: list[StepView]
    reward: float
    input: dict  # Function arguments
    output: Any  # Function return value
```

### Runtime Helpers

```python
from rllm.sdk.session import wrap_with_session_context

# Wrap agent functions with automatic session context
wrapped_fn = wrap_with_session_context(agent_func, tracer_service_name="my-agent")
output, session_uid = wrapped_fn(metadata, *args, **kwargs)
```

## Architecture

```
rllm/sdk/
├── __init__.py              # Public exports
├── protocol.py              # Data models (Trace, StepView, TrajectoryView)
├── decorators.py            # @trajectory decorator
├── shortcuts.py             # session(), get_chat_client()
├── data_process.py          # Trace-to-model-output conversion utilities
├── session/
│   ├── __init__.py          # Session exports, SESSION_BACKEND config
│   ├── base.py              # SessionProtocol, wrap_with_session_context()
│   ├── contextvar.py        # ContextVarSession (default backend)
│   ├── opentelemetry.py     # OpenTelemetrySession (W3C baggage-based)
│   └── session_buffer.py    # SessionBuffer (ephemeral trace storage)
├── chat/
│   ├── __init__.py          # Chat client exports
│   ├── openai.py            # Unified OpenAI chat client (all client types)
│   └── util.py              # Shared utilities for chat clients
├── proxy/
│   ├── __init__.py          # Proxy module exports
│   ├── litellm_callbacks.py # TracingCallback, SamplingParametersCallback
│   ├── litellm_server.py    # LiteLLM server integration
│   ├── metadata_slug.py     # URL metadata encoding/decoding
│   ├── middleware.py        # MetadataRoutingMiddleware (ASGI)
│   └── proxy_manager.py     # Proxy lifecycle management
├── tracers/
│   ├── __init__.py          # Tracer exports
│   ├── base.py              # TracerProtocol
│   ├── memory.py            # InMemorySessionTracer
│   └── sqlite.py            # SqliteTracer
└── store/
    ├── __init__.py          # Store exports
    └── sqlite_store.py      # SQLite trace storage
```

## Design Principles

1. **Minimal API surface**: Simple, focused functions
2. **Context-based**: Uses Python's `contextvars` for automatic propagation
3. **Distributed-ready**: OpenTelemetry backend for cross-process tracing
4. **Pluggable storage**: Supports in-memory, SQLite, or custom backends
5. **Type-safe**: Full type annotations with Pydantic models
6. **Async-native**: First-class async/await support
7. **Proxy-integrated**: Built-in support for LiteLLM proxy routing

# RLLM SDK - Automatic LLM Trace Collection and RL Training

Lightweight SDK for automatic LLM trace collection using session contexts and trajectory decorators.

## Installation

The SDK is part of the `rllm` package:

```python
from rllm.sdk import session, get_chat_client, trajectory
```

## Quick Start

### Basic Session Usage

```python
from rllm.sdk import session, get_chat_client

llm = get_chat_client(api_key="sk-...", model="gpt-4")

# Create a session to track all LLM calls
with session(experiment="v1") as sess:
    response = llm.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}]
    )
    # Access all traces from this session
    print(f"Collected {len(sess.llm_calls)} traces")
```

### Trajectory Decorator

```python
from rllm.sdk import trajectory, get_chat_client_async

llm = get_chat_client_async(api_key="sk-...", model="gpt-4")

@trajectory(name="solver")
async def solve_math_problem(problem: str):
    # Each LLM call automatically becomes a step
    response1 = await llm.chat.completions.create(
        messages=[{"role": "user", "content": f"Solve: {problem}"}]
    )
    response2 = await llm.chat.completions.create(
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

### 1. Session Context
Tracks all LLM calls within a context for debugging and analysis.

```python
from rllm.sdk import session

# Auto-generated session name
with session(experiment="v1") as sess:
    llm.chat.completions.create(...)
    print(sess.llm_calls)  # List of Trace objects
```

### 2. Metadata Inheritance
Nested sessions automatically merge metadata.

```python
with session(experiment="v1"):
    with session(task="math"):
        # All traces get: {experiment: "v1", task: "math"}
        llm.chat.completions.create(...)
```

### 3. Storage Backends
Choose between in-memory or persistent storage.

```python
from rllm.sdk import session
from rllm.sdk.session import SqliteSessionStorage

# In-memory (default)
with session() as sess:
    llm.call()

# SQLite (persistent, multi-process)
storage = SqliteSessionStorage("traces.db")
with session(storage=storage) as sess:
    llm.call()
```

## API Reference

### Core Functions

```python
# Session management
session(**metadata) -> SessionContext  # Auto-generates session name
get_current_session() -> ContextVarSession | None
get_current_metadata() -> dict

# Chat clients
get_chat_client(api_key, model, ...) -> ProxyTrackedChatClient
get_chat_client_async(api_key, model, ...) -> ProxyTrackedAsyncChatClient

# Decorators
@trajectory(name: str, **metadata) -> Callable
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

## Architecture

```
rllm/sdk/
├── __init__.py           # Public exports
├── protocol.py           # Data models (Trace, StepView, TrajectoryView)
├── decorators.py         # @trajectory decorator
├── shortcuts.py          # session(), get_chat_client()
├── session/
│   ├── base.py           # SessionProtocol
│   ├── contextvar.py     # ContextVarSession (default)
│   └── storage.py        # InMemoryStorage, SqliteSessionStorage
├── chat/
│   ├── proxy_chat_client.py    # Proxy-enabled chat client
│   └── simple_chat_client.py   # Simple chat client
├── tracers/
│   ├── base.py           # TracerProtocol
│   ├── memory.py         # InMemorySessionTracer
│   └── sqlite.py         # SqliteTracer
└── store/
    └── sqlite_store.py   # SQLite trace storage

## Design Principles

1. **Minimal API surface**: Simple, focused functions
2. **Context-based**: Uses Python's `contextvars` for automatic propagation
3. **Pluggable storage**: Supports in-memory, SQLite, or custom backends
4. **Type-safe**: Full type annotations with Pydantic models
5. **Async-native**: First-class async/await support

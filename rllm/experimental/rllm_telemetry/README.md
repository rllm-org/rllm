# Rllm Telemetry

Agent observability plugin for [Google ADK](https://github.com/google/adk-python). Captures full agent traces — LLM calls, tool executions, agent hierarchy, and events — and streams them to your Rllm backend.

## Install

```bash
pip install -e ./rllm_telemetry   # local development
# or, once published:
# pip install rllm-telemetry
```

## Quick Start

```python
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from rllm_telemetry import RllmTelemetryPlugin

agent = Agent(name="my_agent", model="gemini-2.0-flash", instruction="You are a helpful assistant.")
runner = Runner(agent=agent, app_name="my_app", session_service=InMemorySessionService())

# Register the plugin — that's it
runner.plugin_manager.register_plugin(
    RllmTelemetryPlugin(api_key="sk-...", endpoint="https://your-backend.com/v1/traces")
)
```

Or use the one-liner:

```python
import rllm_telemetry

rllm_telemetry.instrument(runner, api_key="sk-...", endpoint="https://your-backend.com/v1/traces")
```

## What Gets Captured

The plugin hooks into every stage of the ADK execution lifecycle:

| ADK Callback | Trace Record | Key Data |
|---|---|---|
| `before_run` / `after_run` | **Invocation** | session, user message, agent tree, aggregated token counts |
| `before_agent` / `after_agent` | **AgentSpan** | agent name/type, branch, duration |
| `before_model` / `after_model` | **LlmSpan** | model, system prompt, chat history, tools, token usage |
| `before_tool` / `after_tool` | **ToolSpan** | tool name/type, args, result, duration |
| `on_event` | **EventRecord** | raw ADK event with content, actions, state deltas |

## Trace Hierarchy

Mirrors the ADK execution model:

```
Session                          (lazy — created on first invocation)
└─ Invocation                    (one user message → response cycle)
     ├─ AgentSpan                (one agent.run_async() call)
     │    ├─ LlmSpan            (one generate_content call)
     │    └─ ToolSpan           (one tool execution)
     └─ EventRecord             (each Event yielded by the runner)
```

## Configuration

Pass a `RllmConfig` for full control:

```python
from rllm_telemetry import RllmConfig, RllmTelemetryPlugin

config = RllmConfig(
    api_key="sk-...",
    endpoint="https://your-backend.com/v1/traces",
    capture_content=True,         # capture full prompts & responses (default: True)
    capture_tools=True,           # capture tool args & results (default: True)
    max_content_length=-1,        # truncate strings at N chars (-1 = no limit)
    batch_size=64,                # records per HTTP flush
    flush_interval_seconds=2.0,   # max seconds between flushes
    timeout_seconds=10.0,         # HTTP request timeout
    headers={"X-Custom": "val"},  # extra headers on every request
)

plugin = RllmTelemetryPlugin(config=config)
```

### Privacy mode

To capture trace structure (timing, token counts, tool names) without prompt content:

```python
config = RllmConfig(api_key="sk-...", capture_content=False, capture_tools=False)
```

## Wire Format

Records are sent as newline-delimited JSON (NDJSON) via `POST`:

```jsonl
{"type":"session","data":{"session_id":"...","app_name":"my_app","user_id":"user1","created_at":1709500000.0}}
{"type":"invocation.start","data":{"invocation_id":"e-...","session_id":"...","started_at":1709500000.1,...}}
{"type":"agent.start","data":{"span_id":"...","agent_name":"my_agent","agent_type":"LlmAgent",...}}
{"type":"llm.start","data":{"span_id":"...","request":{"model":"gemini-2.0-flash","system_instruction":"..."},...}}
{"type":"llm.end","data":{"span_id":"...","response":{"usage":{"input_tokens":150,"output_tokens":42}},...}}
{"type":"event","data":{"event_id":"...","author":"my_agent","is_final_response":true,...}}
{"type":"agent.end","data":{"span_id":"...","duration_ms":1234.5,...}}
{"type":"invocation.end","data":{"invocation_id":"e-...","duration_ms":1500.0,"total_input_tokens":150,...}}
```

## Schema Reference

All schemas are Pydantic v2 models importable from `rllm_telemetry.schemas`:

- **`SessionRecord`** — `session_id`, `app_name`, `user_id`, `created_at`
- **`InvocationRecord`** — `invocation_id`, `session_id`, `user_message`, `root_agent` (recursive `AgentInfo`), timing, aggregated metrics (`total_input_tokens`, `total_output_tokens`, `llm_call_count`, `tool_call_count`, `error_count`)
- **`AgentSpanRecord`** — `span_id`, `agent_name`, `agent_type`, `branch`, timing
- **`LlmSpanRecord`** — `span_id`, `agent_name`, `request` (`LlmRequest` with model, system instruction, contents, tools, generation config), `response` (`LlmResponseData` with content, finish reason, token usage, errors), timing
- **`ToolSpanRecord`** — `span_id`, `tool_name`, `tool_type`, `is_long_running`, `args`, `result`, `error`, timing
- **`EventRecord`** — `event_id`, `author`, `content`, `partial`, `is_final_response`, `actions` (state/artifact deltas, transfers, escalation)
- **`TraceEnvelope`** — `{"type": "<span_type>", "data": {...}}` wrapper for NDJSON lines

## Backend Endpoint Contract

Your ingest endpoint should accept:

```
POST /v1/traces
Content-Type: application/x-ndjson
Authorization: Bearer <api_key>

<NDJSON body>
```

Return `2xx` on success. The plugin logs warnings for `4xx`/`5xx` but never crashes the agent.

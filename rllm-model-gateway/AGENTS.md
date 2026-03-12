# rllm-model-gateway

## Overview

A lightweight, standalone Python gateway that sits between RL agents and inference servers (vLLM). It transparently captures every LLM call's token IDs, logprobs, and messages — without requiring any modification to agent code. Agents use standard `OpenAI(base_url=gateway_url)`.

## Development Setup

This package uses `uv` for dependency management and `hatchling` as the build backend. Do not use setuptools or pip.

```bash
cd rllm/rllm-model-gateway
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Set up pre-commit hooks (one-time, from the rllm repo root)
cd .. && pre-commit install && cd rllm-model-gateway
```

To run tests:

```bash
# Unit tests (no external dependencies)
python -m pytest tests/unit/ -x -q

# Integration tests against a real vLLM server (auto-skipped if unreachable)
python -m pytest tests/integration/ -x -v
```

## Package Layout

```
src/rllm_model_gateway/
├── __init__.py           # Public API exports
├── __main__.py           # python -m rllm_model_gateway
├── _version.py           # Package version
├── server.py             # FastAPI app factory + CLI entrypoint
├── proxy.py              # httpx reverse proxy (non-streaming + SSE streaming)
├── middleware.py          # ASGI middleware: session extraction + param injection
├── session_router.py     # Pluggable session-sticky worker routing
├── session_manager.py    # Session CRUD lifecycle
├── data_process.py       # Token/logprob extraction, response sanitization
├── models.py             # Pydantic models (TraceRecord, GatewayConfig, etc.)
├── client.py             # GatewayClient + AsyncGatewayClient
└── store/
    ├── base.py           # TraceStore protocol (6 async methods)
    ├── sqlite_store.py   # SQLite with junction table (default)
    └── memory_store.py   # In-memory (testing)
```

## Key Design Decisions

1. **No LiteLLM** — Direct httpx reverse proxy. Session-sticky routing requires custom logic that LiteLLM can't provide, and httpx handles HTTP forwarding + SSE natively. This keeps deps to 6 packages.

2. **Zero retokenization** — Token IDs come from vLLM's `return_token_ids=True` response field, not from a local tokenizer. The middleware injects this parameter automatically.

3. **Implicit sessions** — First request to `/sessions/{sid}/v1/chat/completions` auto-creates the session. No explicit creation required.

4. **ASGI middleware** — `SessionRoutingMiddleware` operates at the ASGI level (not FastAPI middleware) so it can intercept and rewrite the request body before FastAPI route matching.

5. **Pluggable routing** — `RoutingPolicy` protocol allows swapping the worker selection strategy. Default is `StickyLeastLoadedPolicy` (LRU cache + least-loaded fallback, mirroring verl's pattern).

6. **Pluggable storage** — `TraceStore` protocol (duck-typed, not ABC). SQLite for single-node, MemoryStore for testing, extensible to DynamoDB/PostgreSQL.

## Request Flow

```
Agent → /sessions/{sid}/v1/chat/completions
  → SessionRoutingMiddleware:
      1. Extract session_id from URL, rewrite path to /v1/chat/completions
      2. Inject logprobs=True, return_token_ids=True into body
  → FastAPI route handler (/v1/{path:path}):
      1. SessionManager.ensure_session(sid)
      2. ReverseProxy.handle(request):
          a. SessionRouter.route(session_id) → pick worker
          b. httpx forward to worker
          c. Extract token_ids + logprobs from response
          d. Build TraceRecord, persist to store
          e. Strip vLLM fields from response
          f. Return clean response to agent
```

## Streaming

For SSE streaming requests (`stream=True`):
- Chunks forwarded to agent in real-time via `StreamingResponse`
- Chunks buffered internally for trace assembly
- `prompt_token_ids` extracted from first chunk only
- Delta `token_ids` and logprobs accumulated across chunks
- After `[DONE]`, `build_trace_record_from_chunks()` assembles the trace

## vLLM Response Fields

The gateway strips these vLLM-specific fields before returning to the agent (verified against vLLM 0.11):

| Level | Field | Purpose |
|-------|-------|---------|
| Root | `prompt_token_ids` | Captured for trace, stripped from response |
| Root | `prompt_logprobs` | Not used (always `null` without explicit request) |
| Root | `kv_transfer_params` | Disaggregated prefill feature |
| Choice | `token_ids` | Captured for trace, stripped from response |
| Choice | `stop_reason` | vLLM-specific; standard OpenAI uses `finish_reason` |

## Provenance

| Module | Inspired By | Key Changes |
|--------|------------|-------------|
| `store/sqlite_store.py` | `rllm/sdk/store/sqlite_store.py` | Simplified schema; added `list_sessions()`, `delete_session()` |
| `data_process.py` | `rllm/sdk/data_process.py` | Removed `ModelOutput`/`Step`/`Trajectory` deps; outputs `TraceRecord` |
| `middleware.py` | `rllm/sdk/proxy/middleware.py` + `litellm_callbacks.py` | Merged session routing + param injection; pure ASGI (no LiteLLM) |
| `session_router.py` | verl `agent_loop.py` + miles `router.py` | Combined sticky routing (verl LRU) with health checks (miles pattern) |
| `proxy.py` | miles `router.py` `_do_proxy()` | Added streaming support, trace capture, response sanitization |

## Dependencies

6 runtime dependencies: `fastapi`, `uvicorn`, `httpx`, `pydantic`, `aiosqlite`, `PyYAML`. No torch/ray/verl/transformers.

Build backend: `hatchling`. Do not use setuptools.

## Usage Example

```python
from rllm_model_gateway import GatewayClient

client = GatewayClient("http://localhost:9090")
sid = client.create_session()
url = client.get_session_url(sid)  # → http://localhost:9090/sessions/{sid}/v1

# Agent uses: OpenAI(base_url=url, api_key="EMPTY")
# Training retrieves: client.get_session_traces(sid)
```

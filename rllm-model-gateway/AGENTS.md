# rllm-model-gateway

## Overview

A standalone Python gateway that sits between RL agents/trainers and inference backends. Captures every LLM call's token IDs, logprobs, and messages — without modifying agent code. Agents use standard `OpenAI(base_url=gateway_url)`.

The gateway has two operational modes that can coexist in the same process:

- **Worker pool** — sticky routing across vLLM workers, used by verl/tinker training. The trainer's `GatewayManager` owns the lifecycle (in-process or subprocess) and registers workers via `POST /admin/workers`.
- **Upstream proxy** — provider-agnostic forwarding to OpenAI-compatible URLs (OpenAI, Anthropic, vLLM, etc.), used by eval and sandboxed CLI-agent harnesses. Routes are matched by `body["model"]`.

If neither a route match nor a worker pool is configured, a request returns 404. Routes take precedence; misses fall through to the worker pool.

## Development setup

This package uses `uv` for dependency management and `hatchling` as the build backend. Do not use setuptools or pip.

```bash
cd rllm-model-gateway
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Set up pre-commit hooks (one-time, from the rllm repo root)
cd .. && pre-commit install && cd rllm-model-gateway
```

Tests:

```bash
python -m pytest tests/unit/ -x -q                  # no external deps
python -m pytest tests/integration/ -x -v           # requires vLLM on localhost:4000 (auto-skip otherwise)
```

## Package layout

```
src/rllm_model_gateway/
├── __init__.py           # Public API exports
├── __main__.py           # python -m rllm_model_gateway
├── _version.py
├── server.py             # FastAPI app factory + CLI entrypoint
├── proxy.py              # httpx reverse proxy: worker-pool + upstream-route paths, SSE streaming
├── middleware.py         # ASGI middleware: session/metadata extraction, param injection, inbound auth
├── metadata.py           # RllmMetadata + extract_metadata (header/body/path precedence)
├── session_router.py     # Pluggable session-sticky worker routing
├── session_manager.py    # Session CRUD lifecycle
├── data_process.py       # Token/logprob extraction, response sanitization
├── models.py             # Pydantic models — TraceRecord, GatewayConfig, UpstreamRoute, …
├── client.py             # GatewayClient + AsyncGatewayClient
└── store/
    ├── base.py           # TraceStore protocol
    ├── sqlite_store.py   # SQLite with run/session/trace tables (default)
    └── memory_store.py   # In-memory (testing)
```

## Identifying a session

Three sources, in precedence order, field-by-field:

1. **`X-RLLM-*` headers** (canonical): `X-RLLM-Session-Id`, `X-RLLM-Run-Id`, `X-RLLM-Harness`, `X-RLLM-Step-Id`, `X-RLLM-Parent-Span-Id`, `X-RLLM-Project`, `X-RLLM-Experiment`. Stripped before forwarding upstream.
2. **Body fallback**: `metadata.rllm.{…}` or top-level `rllm.{…}`.
3. **Legacy URL path**: `/sessions/{sid}/v1/...` — preserved exclusively for `GatewayManager`'s training entry points.

Implementation: `metadata.extract_metadata` populates a `RllmMetadata` model into `scope["state"]["rllm_metadata"]`; the proxy reads it when persisting trace records.

## Key design decisions

1. **No litellm.** Direct httpx reverse proxy. Session-sticky routing requires custom logic, and httpx handles HTTP forwarding + SSE natively. Keeps deps to 6 packages. (Note: `rllm/sdk/proxy/litellm_server.py` in the parent repo is a *separate* subsystem still used by the verl/tinker SDK trainer; it is not part of this package.)

2. **Provider-agnostic upstream routing.** `UpstreamRoute` carries a fully-resolved `upstream_url`, literal `Authorization` header, and per-route `drop_params`. The gateway does not know about provider conventions — the caller (rllm) resolves provider knowledge before populating routes.

3. **Header-stamped metadata over URL stuffing.** The legacy `/sessions/{sid}/v1/...` shape conflates routing with identity. New code stamps `X-RLLM-*` headers on requests; old code keeps working because the middleware still recognizes the URL path.

4. **Zero retokenization.** Token IDs come from vLLM's `return_token_ids=True` response field, not from a local tokenizer. The middleware injects this parameter automatically when targeting the worker pool; for upstream routes that don't return token IDs, traces capture only message-level data.

5. **Implicit sessions.** First request to `/sessions/{sid}/v1/...` (or with `X-RLLM-Session-Id`) auto-creates the session.

6. **ASGI middleware.** `SessionRoutingMiddleware` operates at the ASGI level (not FastAPI middleware) so it can intercept and rewrite the request body before FastAPI route matching.

7. **Pluggable routing + storage.** `RoutingPolicy` (worker selection) and `TraceStore` (persistence) are protocols. Default routing policy: `StickyLeastLoadedPolicy` (LRU + least-loaded). Default store: SQLite.

8. **Run lifecycle.** `GatewayConfig.run_id` tags every persisted trace; the run is registered on startup and ended on shutdown so the cross-run viewer can group sessions by gateway lifetime.

9. **Optional inbound auth.** `inbound_auth_token` requires `Authorization: Bearer <token>` on every request. Used when the gateway is exposed via tunnel for remote-sandbox eval.

## Request flow

### Worker-pool path (training)

```
Agent → /sessions/{sid}/v1/chat/completions
  → SessionRoutingMiddleware:
      1. Extract metadata from headers/body/URL → RllmMetadata
      2. Strip /sessions/{sid} prefix → /v1/chat/completions
      3. Inject logprobs=True, return_token_ids=True into body
  → /v1/{path:path} handler:
      1. SessionManager.ensure_session(sid)
      2. ReverseProxy.handle:
          a. body["model"] not in routes → worker-pool path
          b. SessionRouter.route(session_id) → pick worker
          c. httpx forward to worker
          d. Extract token_ids + logprobs from response
          e. Build TraceRecord, persist
          f. Strip vLLM fields, return clean response
```

### Upstream-route path (eval)

```
Harness → /v1/chat/completions  (with X-RLLM-* headers)
  → SessionRoutingMiddleware:
      1. (optional) inbound bearer auth check
      2. Extract metadata from headers → RllmMetadata
  → /v1/{path:path} handler:
      1. SessionManager.ensure_session(sid)
      2. ReverseProxy.handle:
          a. body["model"] in routes → upstream-route path
          b. Strip X-RLLM-* and client auth from forwarded headers
          c. Inject route.auth_header
          d. Drop body params in route.drop_params
          e. httpx forward to route.upstream_url
          f. Build TraceRecord (without token_ids if upstream doesn't return them), persist
          g. Return upstream response unchanged
```

## Streaming

For SSE streaming requests (`stream=True`):

- Chunks forwarded to client in real-time via `StreamingResponse`
- Chunks buffered for trace assembly
- After `[DONE]`, `build_trace_record_from_chunks` assembles the trace and persists asynchronously (does not block the response)

## vLLM response fields

The gateway strips these vLLM-specific fields before returning to the agent (verified against vLLM 0.11):

| Level | Field | Purpose |
|-------|-------|---------|
| Root | `prompt_token_ids` | Captured for trace, stripped |
| Root | `prompt_logprobs` | Not used (always `null` without explicit request) |
| Root | `kv_transfer_params` | Disaggregated prefill feature |
| Choice | `token_ids` | Captured for trace, stripped |
| Choice | `stop_reason` | vLLM-specific; standard OpenAI uses `finish_reason` |

For upstream routes that don't return these fields (OpenAI, Anthropic, etc.), there's nothing to strip.

## Trace store schema

The SQLite store has three tables:

- `runs` — gateway lifetimes, registered on startup via `run_id`. Carries `run_metadata`, start/end timestamps.
- `sessions` — per-session metadata, foreign-keyed to `runs.run_id` (nullable for the unstamped bucket).
- `traces` — `TraceRecord` rows, foreign-keyed to `sessions.session_id`. Indexed on `(run_id, timestamp)` and `(session_id, timestamp)` for cross-run queries.

`MemoryStore` mirrors the same shape with dicts, used for tests.

## Provenance

| Module | Inspired by | Key changes |
|--------|------------|-------------|
| `store/sqlite_store.py` | `rllm/sdk/store/sqlite_store.py` | Added run-table; cross-run queries; junction tables |
| `data_process.py` | `rllm/sdk/data_process.py` | Removed `ModelOutput`/`Step`/`Trajectory` deps; outputs `TraceRecord` |
| `middleware.py` | `rllm/sdk/proxy/middleware.py` + `litellm_callbacks.py` | Merged routing + injection + inbound auth; pure ASGI |
| `session_router.py` | verl `agent_loop.py` + miles `router.py` | Sticky routing (verl LRU) + health checks (miles) |
| `proxy.py` | miles `router.py` `_do_proxy()` | Streaming, trace capture, response sanitization, upstream-route branch |

## Dependencies

6 runtime deps: `fastapi`, `uvicorn`, `httpx`, `pydantic`, `aiosqlite`, `PyYAML`. No torch/ray/verl/transformers.

Build backend: `hatchling`. Do not use setuptools.

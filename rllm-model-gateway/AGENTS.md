# rllm-model-gateway

Lightweight FastAPI gateway sitting between an agent and a single LLM endpoint. Two modes, picked at startup, mutually exclusive:

- **Passthrough**: forward bytes to a configured `upstream_url`. Agent's auth header is stripped; `upstream_api_key` (if set) is injected as the upstream's auth.
- **Adapter**: convert wire request → `NormalizedRequest` → in-process `async def adapter(req) -> NormalizedResponse` → wire response.

Captures every call as a `TraceRecord` in SQLite. Adapter mode optionally captures token-level extras into a separate blob row. Two-tier API key auth: admin key for management endpoints, agent key for proxy endpoints.

## Development setup

```bash
cd rllm/rllm-model-gateway
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

cd .. && pre-commit install && cd rllm-model-gateway

python -m pytest tests/ -q
```

## Package layout

```
src/rllm_model_gateway/
├── __init__.py
├── __main__.py            # python -m rllm_model_gateway
├── _version.py
├── server.py              # FastAPI app factory + CLI; request handler
├── middleware.py          # SessionRoutingMiddleware (extracts sid, rewrites path)
├── auth.py                # AuthMiddleware (two-tier bearer)
├── config.py              # GatewayConfig
├── normalized.py          # NormalizedRequest / NormalizedResponse / Message / ToolCall / ToolSpec / AdapterError
├── extras.py              # pinned key constants for NormalizedResponse.extras
├── trace.py               # TraceRecord + serialize_extras / deserialize_extras
├── client.py              # GatewayClient + AsyncGatewayClient
├── endpoints/
│   ├── __init__.py        # SHAPERS = {"chat_completions": <module>, …}
│   ├── chat_completions.py
│   ├── completions.py
│   ├── responses.py
│   └── anthropic_messages.py
└── store/
    ├── base.py            # TraceStore protocol
    ├── sqlite_store.py    # WAL-mode SQLite, three tables (sessions / traces / trace_extras)
    └── memory_store.py    # in-memory test fixture (importable; not exposed via config)
```

## Endpoint shapers

Each shaper module exposes:

```python
NAME: str                                # e.g. "chat_completions"
PATH: str                                # FastAPI route, e.g. "/v1/chat/completions"
UPSTREAM_PATH: str                       # appended to upstream_url in passthrough

def to_normalized_request(body: dict) -> NormalizedRequest
def parse_upstream_response(body: dict) -> NormalizedResponse        # passthrough
def parse_upstream_stream(chunks: list[dict]) -> NormalizedResponse  # passthrough
def from_normalized_response_nonstream(resp, model: str) -> dict     # adapter
async def from_normalized_response_stream(resp, model: str) -> AsyncIterator[str]  # adapter
```

`PATH` and `UPSTREAM_PATH` differ because the gateway's internal routing path includes `/v1` (so it matches what SDKs send), while the upstream-relative path follows the upstream's SDK convention:

| Endpoint | PATH (gateway-internal) | UPSTREAM_PATH (forwarded) |
|---|---|---|
| chat_completions | `/v1/chat/completions` | `/chat/completions` |
| completions | `/v1/completions` | `/completions` |
| responses | `/v1/responses` | `/responses` |
| anthropic_messages | `/v1/messages` | `/v1/messages` |

So users set `upstream_url` per the upstream's own SDK convention: `https://api.openai.com/v1` (path appended is `/chat/completions` → resolves to `https://api.openai.com/v1/chat/completions`); `https://api.anthropic.com` (path appended is `/v1/messages` → `https://api.anthropic.com/v1/messages`).

Streaming in adapter mode is fake-streamed: the adapter returns a complete `NormalizedResponse`, the gateway emits SSE events post-hoc in the wire format the agent asked for.

## Per-request flow

```
POST /sessions/{sid}/v1/<endpoint>
  → AuthMiddleware                  Bearer-checks admin or agent key (constant-time)
  → SessionRoutingMiddleware        extracts {sid}, rewrites path to /v1/<endpoint>
  → endpoint handler:
      1. Look up session in store; 404 if absent
      2. Merge session sampling_params into request body per
         config.sampling_params_priority
      3. If config.model is set, rewrite body.model
      4. Adapter mode:
           - reject n>1 (unsupported in adapter mode)
           - shaper.to_normalized_request → adapter(req) → shaper.from_normalized_response_*
           - AdapterError → return its (message, status_code)
         Passthrough mode:
           - strip Authorization/x-api-key headers
           - inject upstream_api_key (if set) as Authorization+x-api-key
           - httpx forward to upstream_url + UPSTREAM_PATH (with retry on ConnectError)
      5. Spawn async _do_persist (TraceRecord + extras blob)
      6. Return wire response
```

## Adapter contract

```python
async def adapter(req: NormalizedRequest) -> NormalizedResponse:
    ...
```

- Must be `async def`. Sync functions raise `TypeError` at `create_app` time.
- `NormalizedRequest.session_id` is set by the gateway; adapter can use it as a cache key (worker affinity, prefix-cache hint, etc.).
- Return `NormalizedResponse` populated with whatever subset of fields makes sense.
- Raise `AdapterError(message, status_code=4xx)` to signal a specific HTTP status. Other exceptions become 502.

## Per-trace fields the adapter can populate

| Field | Storage | Purpose |
|---|---|---|
| `extras: dict[str, Any]` | `trace_extras` blob (msgpack) | Large binary-ish data; fetched per-trace. Pinned keys in `extras.py`: `prompt_ids`, `completion_ids`, `logprobs`, `prompt_logprobs`, `routing_matrices`. |
| `metrics: dict[str, float]` | `traces.metrics` JSON column | Numeric measurements you want to aggregate. Returned with `get_session_traces`. Gateway always adds `gateway_latency_ms`. |
| `metadata: dict[str, Any]` | `traces.metadata` JSON column | Free-form non-numeric tags. Returned with `get_session_traces`. |

## Storage schema (SQLite, WAL mode, one file)

```sql
CREATE TABLE sessions (
    session_id      TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    metadata        TEXT,
    sampling_params TEXT
);

CREATE TABLE traces (
    trace_id          TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    timestamp         REAL NOT NULL,
    endpoint          TEXT NOT NULL,
    model             TEXT,
    messages          TEXT,
    prompt            TEXT,
    tools             TEXT,
    sampling_params   TEXT,
    content           TEXT,
    reasoning         TEXT,
    tool_calls        TEXT,
    finish_reason     TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    metrics           TEXT,
    metadata          TEXT
);
CREATE INDEX idx_traces_session ON traces(session_id, timestamp);

CREATE TABLE trace_extras (
    trace_id  TEXT PRIMARY KEY REFERENCES traces(trace_id) ON DELETE CASCADE,
    format    TEXT NOT NULL,    -- "msgpack"
    data      BLOB NOT NULL
);
```

The split between `traces` and `trace_extras` keeps `get_traces(session_id)` free of blob I/O.

## Two-tier auth

`AuthMiddleware` runs ahead of `SessionRoutingMiddleware`. Accepts both
`Authorization: Bearer <key>` and `x-api-key: <key>` (Anthropic SDK convention).

- `/health`: public.
- `/sessions/{sid}/v1/...`: agent OR admin key.
- everything else: admin key only.

Comparison uses `hmac.compare_digest` (constant-time).

## Trace persistence

Always async fire-and-forget. The handler calls `_spawn_persist(...)`, which schedules `_do_persist` on the event loop and tracks the task in a per-app set. `POST /admin/flush` drains the set then issues a SQLite WAL checkpoint. The lifespan exit also drains on shutdown (no timeout — uvicorn waits).

## Dependencies

Runtime: `fastapi`, `uvicorn`, `httpx`, `pydantic`, `aiosqlite`, `PyYAML`, `msgpack`, `openai` (typing only).
Build backend: `hatchling`. Do not use setuptools.

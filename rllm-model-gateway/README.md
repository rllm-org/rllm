# rllm-model-gateway

Lightweight model gateway that sits between agents/trainers and inference backends. Captures every LLM call as a `TraceRecord` — token IDs, logprobs, messages, and rLLM session metadata — with zero modifications to agent code.

Two operational modes, selectable via configuration:

- **Worker pool** — in-process or HTTP forwarding to vLLM workers with session-sticky routing. Used by verl/tinker training via `GatewayManager`.
- **Upstream proxy** — forwards to provider URLs (OpenAI, Anthropic, vLLM, anything OpenAI-compatible) with route-specific auth and per-route param stripping. Used by eval and sandboxed CLI-agent harnesses.

The same gateway can act as both at once: routes are looked up by `body["model"]`; unmatched models fall through to the worker pool.

## Quick start

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# Worker-pool mode (training)
rllm-model-gateway --port 9090 --worker http://localhost:8000/v1

# Upstream-proxy mode (eval) — see "Configuration" below
rllm-model-gateway --config gateway.yaml
```

## Identifying a session

The gateway extracts session identity from one of three sources, in precedence order:

1. **`X-RLLM-*` headers** (preferred — set by harness shims):
   - `X-RLLM-Session-Id` (required for tracing)
   - `X-RLLM-Run-Id`, `X-RLLM-Harness`, `X-RLLM-Step-Id`
   - `X-RLLM-Parent-Span-Id`, `X-RLLM-Project`, `X-RLLM-Experiment`
2. **Request body** — `metadata.rllm.{session_id, run_id, …}` or top-level `rllm.{…}`.
3. **Legacy URL path** — `/sessions/{sid}/v1/...`. Preserved for back-compat with the training proxy entry points (verl/tinker `GatewayManager`).

Precedence is field-by-field — a request can carry `session_id` in the URL and `run_id` in a header, and both contribute. The `X-RLLM-*` headers are gateway-internal and stripped before forwarding upstream.

## Agent side (no rLLM dependency)

```python
from openai import OpenAI

# Header-stamped (preferred for new code)
client = OpenAI(
    base_url="http://localhost:9090/v1",
    api_key="EMPTY",
    default_headers={
        "X-RLLM-Session-Id": session_id,
        "X-RLLM-Run-Id": run_id,
        "X-RLLM-Harness": "claude-code",
    },
)

# Path-stamped (legacy / training)
client = OpenAI(
    base_url=f"http://localhost:9090/sessions/{session_id}/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Works with any OpenAI-compatible agent framework (ADK, Strands, LangChain, OpenAI Agents SDK, etc.).

## Training side

```python
from rllm_model_gateway import GatewayClient

client = GatewayClient("http://localhost:9090")

session_id = client.create_session()
agent_url = client.get_session_url(session_id)
# → "http://localhost:9090/sessions/{session_id}/v1"

# After agent runs, retrieve traces with full token data
traces = client.get_session_traces(session_id)
for trace in traces:
    print(trace.prompt_token_ids)       # From vLLM's return_token_ids
    print(trace.completion_token_ids)
    print(trace.logprobs)
```

## Eval / upstream-proxy mode

Configure routes in YAML; the gateway forwards by model name:

```yaml
host: 0.0.0.0
port: 9090
run_id: "eval-2026-05-03-claude"
run_metadata:
  benchmark: swe-bench-verified
  agent: claude-code
routes:
  "claude-sonnet-4":
    upstream_url: https://api.anthropic.com/v1
    auth_header: "x-api-key ${ANTHROPIC_API_KEY}"
    drop_params: [logprobs, return_token_ids]
  "gpt-4o":
    upstream_url: https://api.openai.com/v1
    auth_header: "Bearer ${OPENAI_API_KEY}"
```

The route's `auth_header` replaces the client's, so harnesses don't need provider keys. `drop_params` strips body fields the upstream rejects (e.g. `max_tokens` for OpenAI o-series).

## Inbound bearer auth

Set `inbound_auth_token` to require `Authorization: Bearer <token>` on every request. Used by remote-sandbox eval, where the gateway is exposed via a public tunnel and the bearer gates access. See `tests/unit/test_inbound_auth.py`.

## Run lifecycle

`GatewayConfig.run_id` (and `run_metadata`) tag every persisted trace with the gateway's lifetime. On startup the run is registered in the store; on shutdown it's marked ended. The cross-run viewer in [rllm-console](../rllm-console/README.md) groups sessions by `run_id`.

## Development

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Set up pre-commit hooks (one-time, from the rllm repo root)
cd .. && pre-commit install && cd rllm-model-gateway

# Unit tests (no external dependencies)
python -m pytest tests/unit/ -x -q

# Integration tests (requires vLLM on localhost:4000, auto-skipped otherwise)
python -m pytest tests/integration/ -x -v
```

## Configuration

### CLI

```bash
rllm-model-gateway \
  --port 9090 \
  --db-path ./traces.db \
  --worker http://vllm-0:8000/v1 \
  --worker http://vllm-1:8000/v1
```

### YAML

```yaml
host: 0.0.0.0
port: 9090
db_path: ~/.rllm/gateway.db
store_worker: sqlite              # or "memory"
sampling_params_priority: client  # or "session"

# Worker pool (training)
workers:
  - url: http://vllm-0:8000/v1
    model_name: Qwen/Qwen2.5-7B-Instruct

# Upstream routes (eval) — matched by body["model"]
routes:
  "Qwen/Qwen2.5-7B-Instruct":
    upstream_url: https://api.openai.com/v1
    auth_header: "Bearer ${OPENAI_API_KEY}"

# Optional run lifecycle (tags every trace)
run_id: "train-2026-05-03"
run_metadata:
  experiment: ablation-7

# Optional inbound auth (require Bearer on every request)
inbound_auth_token: ${RLLM_GATEWAY_INBOUND_TOKEN}
```

### Environment variables

`RLLM_GATEWAY_HOST`, `RLLM_GATEWAY_PORT`, `RLLM_GATEWAY_DB_PATH`, `RLLM_GATEWAY_LOG_LEVEL`, `RLLM_GATEWAY_STORE`

## Embedded usage

```python
from rllm_model_gateway import create_app, GatewayConfig

config = GatewayConfig(
    port=9090,
    workers=[...],     # worker-pool mode
    routes={...},      # upstream-proxy mode (additive — same gateway can do both)
    run_id="my-run",
)
app = create_app(config)
```

## Dynamic worker registration

Workers can be added at runtime via the admin API — useful for verl integration where vLLM addresses are only known after initialization:

```python
client = GatewayClient("http://localhost:9090")
client.add_worker(url="http://vllm-worker-0:8000/v1", model_name="Qwen/Qwen2.5-7B")
```

## Features

- **Two-mode**: worker-pool routing (training) + upstream-proxy routing (eval), composable
- **Header-stamped metadata**: rLLM identity carried via `X-RLLM-*` headers; never leaks upstream
- **Zero retokenization**: token IDs come from vLLM's `return_token_ids=True` response field
- **Run-tagged trace store**: SQLite (default) or in-memory; sessions queryable by `run_id` for cross-run analysis
- **Streaming**: SSE chunk forwarding with trace assembly post-`[DONE]`
- **Inbound bearer auth**: optional, for tunneled deployments
- **Lightweight**: 6 runtime deps, no torch/ray/verl/transformers

## API overview

| Endpoint | Description |
|----------|-------------|
| `POST /v1/{path}` | Proxy with header- or body-stamped session id |
| `POST /sessions/{sid}/v1/{path}` | Legacy path-stamped proxy (training) |
| `POST /sessions` | Create session with metadata |
| `GET /sessions` | List sessions (filterable by `since`, `limit`) |
| `GET /sessions/{sid}` | Session info |
| `GET /sessions/{sid}/traces` | Traces for a session |
| `DELETE /sessions/{sid}` | Delete session and traces |
| `GET /traces/{trace_id}` | Single trace |
| `POST /traces/query` | Cross-session trace fetch |
| `POST /admin/workers` | Register a vLLM worker |
| `DELETE /admin/workers/{id}` | Remove worker |
| `POST /admin/flush` | Drain pending trace writes |
| `POST /admin/reload` | Hot-reload routes/workers from config |
| `GET /health` | Liveness check |

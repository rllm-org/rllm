# rllm-model-gateway

Lightweight model gateway for capturing LLM call traces during RL agent training. Sits between agents and inference servers (vLLM), transparently recording token IDs, logprobs, and conversation data — with zero modifications to agent code.

## Quick Start

```bash
# Create a uv environment
uv venv --python 3.11
source .venv/bin/activate

# Install
uv pip install -e .

# Set up pre-commit hooks (one-time, from the rllm repo root)
cd .. && pre-commit install && cd rllm-model-gateway

# Start with a vLLM worker
rllm-model-gateway --port 9090 --worker http://localhost:8000/v1

# Or with a config file
rllm-model-gateway --config gateway.yaml
```

## Agent Side (Zero rLLM Dependencies)

```python
from openai import OpenAI

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

## Training Side

```python
from rllm_model_gateway import GatewayClient

client = GatewayClient("http://localhost:9090")

# Create session and get URL for the agent
session_id = client.create_session()
agent_url = client.get_session_url(session_id)
# → "http://localhost:9090/sessions/{session_id}/v1"

# After agent runs, retrieve traces with full token data
traces = client.get_session_traces(session_id)
for trace in traces:
    print(trace.prompt_token_ids)       # From vLLM's return_token_ids
    print(trace.completion_token_ids)   # Per-token IDs, no retokenization needed
    print(trace.logprobs)               # Per-token logprobs
```

## Features

- **Zero agent coupling** — Agents use standard `OpenAI(base_url=...)`, no rLLM imports
- **Zero retokenization** — Token IDs captured directly from vLLM responses
- **Partial rollout recovery** — Traces persisted per-call, survive agent crashes
- **Session-sticky routing** — Multi-turn sessions routed to the same worker for prefix caching
- **Streaming support** — SSE streaming with real-time chunk forwarding and trace assembly
- **Pluggable storage** — SQLite (default), in-memory (testing), extensible to DynamoDB/PostgreSQL
- **Lightweight** — 6 dependencies, no torch/ray/verl/transformers

## Development

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Unit tests
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

### YAML (`--config gateway.yaml`)

```yaml
host: "0.0.0.0"
port: 9090
db_path: "~/.rllm/gateway.db"

workers:
  - url: "http://vllm-0:8000/v1"
    model_name: "Qwen/Qwen2.5-7B-Instruct"
  - url: "http://vllm-1:8000/v1"
    model_name: "Qwen/Qwen2.5-7B-Instruct"
```

### Environment Variables

`RLLM_GATEWAY_HOST`, `RLLM_GATEWAY_PORT`, `RLLM_GATEWAY_DB_PATH`, `RLLM_GATEWAY_LOG_LEVEL`, `RLLM_GATEWAY_STORE`

## Embedded Usage

```python
from rllm_model_gateway import create_app, GatewayConfig

config = GatewayConfig(port=9090, workers=[...])
app = create_app(config)

import threading, uvicorn
threading.Thread(target=uvicorn.run, args=(app,), kwargs={"port": 9090}, daemon=True).start()
```

## Dynamic Worker Registration

Workers can be added at runtime via the admin API — useful for verl integration where vLLM addresses are only known after initialization:

```python
client = GatewayClient("http://localhost:9090")
client.add_worker(url="http://vllm-worker-0:8000/v1", model_name="Qwen/Qwen2.5-7B")
```

## API Overview

| Endpoint | Description |
|----------|-------------|
| `POST /sessions/{sid}/v1/chat/completions` | Proxy (agent-facing, OpenAI-compatible) |
| `POST /sessions` | Create session with metadata |
| `GET /sessions/{sid}/traces` | Retrieve traces for a session |
| `POST /admin/workers` | Register a worker |
| `GET /health` | Gateway health check |

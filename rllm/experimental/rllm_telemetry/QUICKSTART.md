# rllm_telemetry — Quick Start

Add real-time observability to your ADK agent in 3 steps.

## 1. Install

```bash
pip install "rllm-telemetry @ git+https://github.com/boredbichon67/rllm.git@dev-rllm_telemetry#subdirectory=rllm/experimental/rllm_telemetry"
```

## 2. Get your API key

1. Go to **https://believable-ambition-rllm-staging.up.railway.app**
2. **Sign up** for an account
3. Navigate to **Settings → API Key → Regenerate**
4. Copy the key and set it as an environment variable:

```bash
export RLLM_API_KEY=rllm_<your-key>
```

## 3. Add one line to your agent code

```python
import rllm_telemetry
from google.adk.runners import InMemoryRunner

agent = ...  # your ADK agent
runner = InMemoryRunner(agent=agent, app_name="my_app")

plugin = rllm_telemetry.instrument(runner)  # ← this is it

# Run your agent as usual
# ...

# When done, close the plugin to mark the session as completed
await plugin.close()
```

That's it. Every LLM call, tool execution, and agent step is now streamed to the observability dashboard at:

**https://believable-ambition-rllm-staging.up.railway.app/observability**

Make sure to select the **ClickHouse** data source tab to see your live spans (not "Imported").

## What you'll see

- **Live session timeline** — each agent run appears as a session with a full span tree
- **LLM calls** — model, tokens, latency, full request/response content
- **Tool calls** — tool name, args, results, duration
- **Dashboard** — aggregate metrics, top models, top tools, error rates

## Options

```python
rllm_telemetry.instrument(
    runner,
    backend="stdout",           # also print spans to terminal (default)
    agent_session_name="my-run", # custom session name in the UI
)
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ValueError: An API key is required` | Set `RLLM_API_KEY` env var |
| `401 Unauthorized` | Check your API key is valid (Settings → API Key) |
| Spans don't appear | Check the ClickHouse data source tab in Observability |

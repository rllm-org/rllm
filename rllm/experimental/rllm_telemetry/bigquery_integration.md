# BigQuery Integration

Stream agent span telemetry directly to Google BigQuery for analytics, dashboards, and long-term storage.

## Prerequisites

- A Google Cloud project with BigQuery enabled
- `google-cloud-bigquery` Python package (`pip install google-cloud-bigquery`)
- Application Default Credentials configured (`gcloud auth application-default login`)

## Quick Start

```python
from rllm_telemetry import RllmConfig, RllmTelemetryPlugin

config = RllmConfig(
    backend="bigquery",
    bq_project="my-gcp-project",
    bq_dataset="agent_traces",
    bq_table="rllm_traces",       # default
    bq_auto_create=True,          # create dataset + table if missing
)

plugin = RllmTelemetryPlugin(config=config)
runner.plugin_manager.register_plugin(plugin)
```

Or with the one-liner:

```python
import rllm_telemetry

rllm_telemetry.instrument(
    runner,
    backend="bigquery",
    bq_project="my-gcp-project",
    bq_dataset="agent_traces",
    bq_auto_create=True,
)
```

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | `str` | `"stdout"` | Set to `"bigquery"` to enable BQ export |
| `bq_project` | `str` | `""` | Google Cloud project ID |
| `bq_dataset` | `str` | `""` | BigQuery dataset name |
| `bq_table` | `str` | `"rllm_traces"` | Table name within the dataset |
| `bq_auto_create` | `bool` | `False` | Auto-create dataset and table if they don't exist |
| `batch_size` | `int` | `64` | Records buffered before flushing to BQ |
| `flush_interval_seconds` | `float` | `2.0` | Max seconds between flushes |

When `bq_auto_create=False` (the default), the exporter raises `BigQueryValidationError` on startup if the dataset or table is missing. Set to `True` for zero-setup development.

## Table Schema

All span types are written to a **single table** using a hybrid schema: promoted columns for frequently-queried fields, plus a `data` JSON column containing the full raw payload.

### `rllm_traces` table

| Column | Type | Mode | Description |
|---|---|---|---|
| `id` | `STRING` | `REQUIRED` | Primary ID — extracted per span type (see [ID extraction](#id-extraction)) |
| `span_type` | `STRING` | `REQUIRED` | Span discriminator (e.g. `llm.end`, `tool.start`, `session`) |
| `session_id` | `STRING` | `REQUIRED` | Agent session ID |
| `invocation_id` | `STRING` | `NULLABLE` | Invocation ID (all span types except `session`) |
| `span_id` | `STRING` | `NULLABLE` | Span ID for `agent.*`, `llm.*`, `tool.*` spans |
| `agent_name` | `STRING` | `NULLABLE` | Name of the agent that produced the span |
| `model` | `STRING` | `NULLABLE` | LLM model name (e.g. `gemini-2.0-flash`) — populated on `llm.*` spans |
| `tool_name` | `STRING` | `NULLABLE` | Tool name — populated on `tool.*` spans |
| `duration_ms` | `FLOAT` | `NULLABLE` | Span duration in milliseconds |
| `input_tokens` | `INTEGER` | `NULLABLE` | LLM input token count |
| `output_tokens` | `INTEGER` | `NULLABLE` | LLM output token count |
| `error` | `STRING` | `NULLABLE` | Error message if the span errored |
| `started_at` | `TIMESTAMP` | `NULLABLE` | Span start time (UTC) |
| `ended_at` | `TIMESTAMP` | `NULLABLE` | Span end time (UTC) |
| `ingested_at` | `TIMESTAMP` | `REQUIRED` | Row insertion time (UTC) — **partition key** |
| `data` | `JSON` | `REQUIRED` | Full raw span payload as JSON |

**Partitioning:** `DAY` on `ingested_at`
**Clustering:** `(span_type, session_id)`

### ID Extraction

The `id` column is populated differently depending on `span_type`:

| `span_type` pattern | `id` source field |
|---|---|
| `session` | `session_id` |
| `invocation.*` | `invocation_id` |
| `agent.*`, `llm.*`, `tool.*` | `span_id` |
| `event` | `event_id` |
| `experiment.case` | `case_id` |
| `experiment.*` | `experiment_id` |
| `trajectory.*` | `trajectory_uid` or `span_id` |

Falls back to a random UUID if the expected field is missing.

## What Lands in the `data` Column

The `data` JSON column contains the full span payload. The exact shape depends on `span_type`. Below are the schemas for each span type relevant to agent span streaming.

### `session`

```json
{
  "session_id": "string",
  "app_name": "string",
  "user_id": "string",
  "created_at": 1709500000.0
}
```

### `invocation.start` / `invocation.end`

```json
{
  "invocation_id": "string",
  "session_id": "string",
  "app_name": "string",
  "user_id": "string",
  "user_message": { },
  "root_agent": {
    "name": "string",
    "description": "string | null",
    "type": "LlmAgent",
    "sub_agents": [ ]
  },
  "started_at": 1709500000.0,
  "ended_at": 1709500001.5,
  "duration_ms": 1500.0,
  "total_input_tokens": 150,
  "total_output_tokens": 42,
  "llm_call_count": 1,
  "tool_call_count": 0,
  "error_count": 0
}
```

Fields `ended_at`, `duration_ms`, and aggregated metrics are `null` on `invocation.start` and populated on `invocation.end`.

### `agent.start` / `agent.end`

```json
{
  "span_id": "string",
  "invocation_id": "string",
  "session_id": "string",
  "agent_name": "planner",
  "agent_description": "string | null",
  "agent_type": "LlmAgent",
  "branch": "root.planner",
  "started_at": 1709500000.0,
  "ended_at": 1709500001.2,
  "duration_ms": 1200.0,
  "output": { },
  "error": null
}
```

### `llm.start` / `llm.end`

```json
{
  "span_id": "string",
  "invocation_id": "string",
  "session_id": "string",
  "agent_name": "planner",
  "request": {
    "model": "gemini-2.0-flash",
    "system_instruction": "You are a helpful assistant.",
    "contents": [ { "role": "user", "parts": [...] } ],
    "tools": [ { "name": "search", "description": "..." } ],
    "generation_config": {
      "temperature": 0.7,
      "top_p": 0.95,
      "top_k": 40,
      "max_output_tokens": 1024,
      "stop_sequences": null,
      "candidate_count": null
    }
  },
  "response": {
    "content": { "role": "model", "parts": [...] },
    "finish_reason": "STOP",
    "model_version": "gemini-2.0-flash-001",
    "error_code": null,
    "error_message": null,
    "usage": {
      "input_tokens": 150,
      "output_tokens": 42,
      "total_tokens": 192,
      "cached_content_token_count": null
    }
  },
  "started_at": 1709500000.0,
  "ended_at": 1709500000.8,
  "duration_ms": 800.0
}
```

`request` is populated on `llm.start`. `response` is `null` on `llm.start` and populated on `llm.end`.

### `tool.start` / `tool.end`

```json
{
  "span_id": "string",
  "invocation_id": "string",
  "session_id": "string",
  "agent_name": "planner",
  "tool_name": "search_web",
  "tool_description": "Search the web for information",
  "tool_type": "FunctionTool",
  "is_long_running": false,
  "args": { "query": "latest news" },
  "result": { "results": [...] },
  "error": null,
  "started_at": 1709500000.0,
  "ended_at": 1709500000.5,
  "duration_ms": 500.0
}
```

`result` is `null` on `tool.start` and populated on `tool.end`.

### `event`

```json
{
  "event_id": "string",
  "invocation_id": "string",
  "session_id": "string",
  "author": "planner",
  "timestamp": 1709500000.0,
  "branch": "root.planner",
  "content": { "role": "model", "parts": [...] },
  "partial": false,
  "is_final_response": true,
  "actions": {
    "state_delta": { "key": "value" },
    "artifact_delta": { "report.pdf": 1 },
    "transfer_to_agent": null,
    "escalate": null,
    "end_of_agent": null
  },
  "error_code": null,
  "error_message": null
}
```

## Promoted Column Extraction

The exporter extracts promoted columns from the raw span data before inserting:

| Promoted Column | Extraction Logic |
|---|---|
| `session_id` | `data["session_id"]` |
| `invocation_id` | `data["invocation_id"]` |
| `span_id` | `data["span_id"]` |
| `agent_name` | `data["agent_name"]` |
| `model` | `data["request"]["model"]` or `data["model"]` |
| `tool_name` | `data["tool_name"]` |
| `duration_ms` | `data["duration_ms"]` |
| `input_tokens` | `data["response"]["usage"]["input_tokens"]` or `data["input_tokens"]` |
| `output_tokens` | `data["response"]["usage"]["output_tokens"]` or `data["output_tokens"]` |
| `error` | `data["error"]` |
| `started_at` | epoch float → ISO 8601 UTC timestamp |
| `ended_at` | epoch float → ISO 8601 UTC timestamp |
| `ingested_at` | Set at insertion time (current UTC) |

Null values are stripped from the row before insert — BigQuery handles missing nullable fields.

## GCP Console Setup (Copy-Paste SQL)

Only **one table** is needed to support agent span streaming. Run the following in your BigQuery console (GCP Console > BigQuery > SQL Workspace).

Replace `YOUR_PROJECT`, `YOUR_DATASET` with your values. The default dataset is `agent_traces` and default table is `rllm_traces`.

### Step 1 — Create the dataset

```sql
CREATE SCHEMA IF NOT EXISTS `YOUR_PROJECT.YOUR_DATASET`
OPTIONS (
  location = 'US'   -- change to your preferred region
);
```

### Step 2 — Create the spans table

```sql
CREATE TABLE IF NOT EXISTS `YOUR_PROJECT.YOUR_DATASET.rllm_traces` (
  id            STRING      NOT NULL,
  span_type     STRING      NOT NULL,
  session_id    STRING      NOT NULL,
  invocation_id STRING,
  span_id       STRING,
  agent_name    STRING,
  model         STRING,
  tool_name     STRING,
  duration_ms   FLOAT64,
  input_tokens  INT64,
  output_tokens INT64,
  error         STRING,
  started_at    TIMESTAMP,
  ended_at      TIMESTAMP,
  ingested_at   TIMESTAMP   NOT NULL,
  data          JSON        NOT NULL
)
PARTITION BY DATE(ingested_at)
CLUSTER BY span_type, session_id
OPTIONS (
  description = 'rllm_telemetry agent span traces'
);
```

That's it — these two queries create everything needed. Then configure the exporter:

```python
config = RllmConfig(
    backend="bigquery",
    bq_project="YOUR_PROJECT",
    bq_dataset="YOUR_DATASET",
    bq_table="rllm_traces",
    bq_auto_create=False,  # table already exists
)
```

> **Tip:** If you prefer zero-setup, skip the SQL above and set `bq_auto_create=True` — the exporter will create both the dataset and table automatically on first run.

## Example Queries

### Sessions overview

```sql
SELECT
  session_id,
  MIN(started_at) AS first_span,
  MAX(ended_at)   AS last_span,
  COUNT(*)        AS total_spans,
  COUNTIF(span_type LIKE 'llm.%')  AS llm_spans,
  COUNTIF(span_type LIKE 'tool.%') AS tool_spans,
  COUNTIF(error IS NOT NULL)       AS errors
FROM `project.dataset.rllm_traces`
WHERE ingested_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
GROUP BY session_id
ORDER BY first_span DESC
LIMIT 50;
```

### Token usage by model

```sql
SELECT
  model,
  COUNT(*)              AS calls,
  SUM(input_tokens)     AS total_input,
  SUM(output_tokens)    AS total_output,
  AVG(duration_ms)      AS avg_latency_ms
FROM `project.dataset.rllm_traces`
WHERE span_type = 'llm.end'
  AND ingested_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
GROUP BY model
ORDER BY calls DESC;
```

### Tool usage and latency

```sql
SELECT
  tool_name,
  COUNT(*)          AS executions,
  AVG(duration_ms)  AS avg_ms,
  MAX(duration_ms)  AS p100_ms,
  COUNTIF(error IS NOT NULL) AS errors
FROM `project.dataset.rllm_traces`
WHERE span_type = 'tool.end'
  AND ingested_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
GROUP BY tool_name
ORDER BY executions DESC;
```

### Full trace for a session (ordered timeline)

```sql
SELECT span_type, agent_name, tool_name, model, duration_ms, error, started_at, ended_at
FROM `project.dataset.rllm_traces`
WHERE session_id = 'your-session-id'
ORDER BY started_at ASC;
```

### Extract nested fields from JSON data column

```sql
SELECT
  span_type,
  JSON_VALUE(data, '$.agent_name') AS agent,
  JSON_VALUE(data, '$.request.model') AS model,
  JSON_VALUE(data, '$.response.finish_reason') AS finish_reason,
  CAST(JSON_VALUE(data, '$.response.usage.input_tokens') AS INT64) AS input_tokens
FROM `project.dataset.rllm_traces`
WHERE span_type = 'llm.end'
  AND session_id = 'your-session-id';
```

## Combining with Agent Trajectory Streaming

You can use BigQuery as the storage backend while also streaming spans to the rllm-ui backend for real-time visualization:

```python
config = RllmConfig(
    backend="bigquery",
    bq_project="my-gcp-project",
    bq_dataset="agent_traces",
    bq_auto_create=True,

    # Also stream to rllm-ui for live visualization
    agent_endpoint="http://localhost:8000",
    agent_api_key="sk-...",
)
```

This wraps the `BigQueryExporter` with `AgentSpanExporter`, which sends every span to both BigQuery (batched) and the rllm-ui backend (per-record, real-time).

## Error Handling

- The exporter **never crashes the agent**. All BQ errors are logged as warnings.
- If the table is deleted mid-run and `bq_auto_create=True`, the exporter auto-recreates it on the next flush.
- If `bq_auto_create=False` and the table/dataset is missing, `BigQueryValidationError` is raised at startup.
- Streaming insert errors are logged but do not block subsequent flushes.

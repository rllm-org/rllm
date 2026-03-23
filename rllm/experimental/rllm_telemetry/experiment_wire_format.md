# Experiment Wire Format — Backend Storage Guide

This document describes the NDJSON wire format for experiment results streamed from `rllm_telemetry`. It is intended for **UI/backend developers** implementing the storage and query layer.

## Overview

Experiments reuse the same NDJSON transport as observability traces. Each line is a JSON object (a `TraceEnvelope`) with a `type` discriminator and a `data` payload:

```json
{"type": "<span_type>", "data": { ... }}
```

Experiment records use three span types:

| Span Type | When Emitted | Payload Schema |
|---|---|---|
| `experiment.start` | Before the first test case runs | `ExperimentRecord` (partial — no `ended_at`, `summary`) |
| `experiment.case` | After each test case completes | `ExperimentCaseRecord` |
| `experiment.end` | After all cases finish | `ExperimentRecord` (complete — includes `summary`) |

## Streaming Lifecycle

```
experiment.start   ← open the experiment; allocate storage
experiment.case    ← case 1
experiment.case    ← case 2
...
experiment.case    ← case N
experiment.end     ← close the experiment; write summary
```

The `experiment_id` field is consistent across all three record types for a given run.  Cases arrive **in execution order**, not necessarily in dataset order.

## Schemas

### `experiment.start` / `experiment.end` — ExperimentRecord

```jsonc
{
  "type": "experiment.start",  // or "experiment.end"
  "data": {
    "experiment_id": "exp_a1b2c3d4",     // UUID, unique per run
    "name": "Detailed Prompt",            // human-readable experiment name
    "created_at": 1709654400.123,         // Unix epoch seconds (float)
    "ended_at": 1709654412.456,           // null on start, populated on end
    "duration_ms": 12333.0,              // null on start, populated on end
    "summary": {                          // null on start, populated on end
      "scores": {
        "Factuality": 0.82,
        "ExactMatch": 0.60
      },
      "total_cases": 5,
      "error_count": 0,
      "total_duration_ms": 12333.0
    },
    "metadata": {                         // arbitrary user-provided metadata
      "model": "anthropic/claude-haiku-4-5-20251001",
      "prompt_style": "detailed"
    }
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `experiment_id` | `string` | Primary key. UUID generated client-side. |
| `name` | `string` | Display name. Not unique — multiple runs can share the same name. |
| `created_at` | `float` | Unix epoch seconds. |
| `ended_at` | `float \| null` | Null in `experiment.start`, set in `experiment.end`. |
| `duration_ms` | `float \| null` | Wall-clock duration. Null in `experiment.start`. |
| `summary` | `ExperimentSummary \| null` | Null in `experiment.start`. See below. |
| `metadata` | `object` | Free-form key/value pairs (model name, prompt version, git SHA, etc). |

### `experiment.case` — ExperimentCaseRecord

```jsonc
{
  "type": "experiment.case",
  "data": {
    "case_id": "case_x7y8z9",            // UUID, unique per case
    "experiment_id": "exp_a1b2c3d4",     // foreign key to ExperimentRecord
    "input": "What is the population density of France?",
    "expected": "About 105 people per km².",
    "output": "France has approximately 105.2 people per km²...",
    "scores": [
      {
        "name": "Factuality",
        "value": 0.9,
        "metadata": {
          "reasoning": "The answer correctly computes population density..."
        }
      }
    ],
    "error": null,                        // string if the task function raised
    "duration_ms": 2340.5
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `case_id` | `string` | Primary key. UUID generated client-side. |
| `experiment_id` | `string` | Foreign key to the parent experiment. |
| `input` | `any` | The test case input (usually a string, can be structured). |
| `expected` | `any \| null` | Ground truth / reference answer. Null if not provided. |
| `output` | `any \| null` | The task function's return value. Null if the task errored. |
| `scores` | `ScoreRecord[]` | One entry per scorer. May be empty if all scorers failed. |
| `error` | `string \| null` | Error message if the task function or a scorer raised an exception. |
| `duration_ms` | `float \| null` | Time to execute the task function + all scorers for this case. |

### ScoreRecord (embedded in `experiment.case`)

| Field | Type | Notes |
|---|---|---|
| `name` | `string` | Scorer name (e.g. `"ExactMatch"`, `"Factuality"`, `"LlmJudge"`). |
| `value` | `float` | Score in `[0.0, 1.0]`. |
| `metadata` | `object` | Scorer-specific metadata. For `LlmJudge`, contains `reasoning`. |

### ExperimentSummary (embedded in `experiment.end`)

| Field | Type | Notes |
|---|---|---|
| `scores` | `dict[string, float]` | Scorer name to mean score across all cases. |
| `total_cases` | `int` | Total number of cases executed. |
| `error_count` | `int` | Number of cases where `error` is non-null. |
| `total_duration_ms` | `float` | Sum of all case `duration_ms` values. |

## Recommended Storage Schema

### Tables

**`experiments`**
```sql
CREATE TABLE experiments (
    experiment_id   TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    created_at      REAL NOT NULL,
    ended_at        REAL,
    duration_ms     REAL,
    summary         JSONB,          -- ExperimentSummary as JSON
    metadata        JSONB,
    -- status derived: 'running' if ended_at IS NULL, else 'completed'
);
```

**`experiment_cases`**
```sql
CREATE TABLE experiment_cases (
    case_id         TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL REFERENCES experiments(experiment_id),
    input           JSONB NOT NULL,
    expected        JSONB,
    output          JSONB,
    scores          JSONB NOT NULL, -- array of ScoreRecord
    error           TEXT,
    duration_ms     REAL,
);

CREATE INDEX idx_cases_experiment ON experiment_cases(experiment_id);
```

### Ingestion Logic

When processing the NDJSON stream:

1. **`experiment.start`** — `INSERT` into `experiments`. Set `ended_at = NULL`, `summary = NULL`. The experiment is now in `running` state.

2. **`experiment.case`** — `INSERT` into `experiment_cases`. The `experiment_id` foreign key links to the parent. Cases can be displayed in the UI as they arrive (real-time progress).

3. **`experiment.end`** — `UPDATE experiments SET ended_at = ..., duration_ms = ..., summary = ... WHERE experiment_id = ...`. The experiment is now `completed`.

### Handling Edge Cases

- **Incomplete experiments**: If the client crashes, you will receive `experiment.start` and some `experiment.case` records but no `experiment.end`. Detect this via `ended_at IS NULL` and either show as "in progress" or mark as "failed" after a timeout.
- **Duplicate delivery**: The same `experiment_id` or `case_id` may arrive more than once if the client retries. Use `INSERT ... ON CONFLICT DO NOTHING` or upsert semantics.
- **Out-of-order cases**: Cases are emitted as they complete. If the task runs concurrently (e.g. `AsyncEval`), cases may arrive out of dataset order. Store insertion order or `duration_ms` for sorting if needed.

## Relationship to Observability Traces

Experiment records and observability traces (sessions, invocations, LLM spans, tool spans, events) share the **same NDJSON transport and `TraceEnvelope` format**. They arrive on the same HTTP endpoint.

The backend should route records by `type`:
- `type` starts with `experiment.` — route to experiment storage
- All other `type` values — route to trace storage

If an agent is instrumented with both `rllm_telemetry.instrument()` (observability) and `Eval()`/`AsyncEval()` (experiments), the stream will interleave both types of records. They are independent — experiment records do not reference invocation IDs or session IDs.

## Example: Full NDJSON Stream

```
{"type":"experiment.start","data":{"experiment_id":"exp_001","name":"Baseline","created_at":1709654400.0,"metadata":{"model":"claude-haiku"}}}
{"type":"experiment.case","data":{"case_id":"c_1","experiment_id":"exp_001","input":"2+2?","expected":"4","output":"4","scores":[{"name":"ExactMatch","value":1.0,"metadata":{}}],"duration_ms":120.5}}
{"type":"experiment.case","data":{"case_id":"c_2","experiment_id":"exp_001","input":"Capital of France?","expected":"Paris","output":"Paris","scores":[{"name":"ExactMatch","value":1.0,"metadata":{}}],"duration_ms":95.3}}
{"type":"experiment.end","data":{"experiment_id":"exp_001","name":"Baseline","created_at":1709654400.0,"ended_at":1709654401.2,"duration_ms":1200.0,"summary":{"scores":{"ExactMatch":1.0},"total_cases":2,"error_count":0,"total_duration_ms":215.8},"metadata":{"model":"claude-haiku"}}}
```

# RLLM Telemetry — Event Schema Reference

All telemetry events are sent as newline-delimited JSON (NDJSON). Each line is a **TraceEnvelope** containing a `type` discriminator and a `data` payload.

## Wire Format

```json
{"type": "<SpanType>", "data": { ... }}
```

### SpanType values

| Value              | Payload Schema     |
| ------------------ | ------------------ |
| `session`          | SessionRecord      |
| `invocation.start` | InvocationRecord   |
| `invocation.end`   | InvocationRecord   |
| `agent.start`      | AgentSpanRecord    |
| `agent.end`        | AgentSpanRecord    |
| `llm.start`        | LlmSpanRecord      |
| `llm.end`          | LlmSpanRecord      |
| `tool.start`       | ToolSpanRecord     |
| `tool.end`         | ToolSpanRecord     |
| `event`            | EventRecord        |

---

## Hierarchy

```
Session            — long-lived conversation between a user and an app
└─ Invocation      — one user message → final response cycle
     ├─ AgentSpan  — one agent.run_async() call
     │    ├─ LlmSpan   — one LLM generate_content call
     │    └─ ToolSpan  — one tool execution
     └─ EventRecord — each Event yielded by the runner
```

---

## TraceEnvelope

Wrapper sent over the wire. One JSON line per record.

```json
{
  "type": "string",   // one of SpanType values above
  "data": {}          // payload object (see schemas below)
}
```

---

## SessionRecord

Lazily created on the first invocation for a given session. There is no separate "create session" event — the plugin registers the session automatically.

```json
{
  "session_id": "string",
  "app_name": "string",
  "user_id": "string",
  "created_at": 1234567890.123
}
```

| Field        | Type   | Required | Description                          |
| ------------ | ------ | -------- | ------------------------------------ |
| `session_id` | string | yes      | Unique session identifier            |
| `app_name`   | string | yes      | Application name                     |
| `user_id`    | string | yes      | User identifier                      |
| `created_at` | float  | yes      | Unix timestamp of session creation   |

---

## InvocationRecord

One user-message to final-response cycle. Opened by `before_run_callback`, closed by `after_run_callback`.

```json
{
  "invocation_id": "string",
  "session_id": "string",
  "app_name": "string",
  "user_id": "string",
  "user_message": {},
  "root_agent": {},
  "started_at": 1234567890.123,
  "ended_at": 1234567890.456,
  "duration_ms": 333.0,
  "total_input_tokens": 0,
  "total_output_tokens": 0,
  "llm_call_count": 0,
  "tool_call_count": 0,
  "error_count": 0
}
```

| Field                | Type                        | Required | Description                                       |
| -------------------- | --------------------------- | -------- | ------------------------------------------------- |
| `invocation_id`      | string                      | yes      | Unique invocation identifier                      |
| `session_id`         | string                      | yes      | Parent session identifier                         |
| `app_name`           | string                      | yes      | Application name                                  |
| `user_id`            | string                      | yes      | User identifier                                   |
| `user_message`       | object \| null              | no       | Serialised `Content` of the user's input          |
| `root_agent`         | [AgentInfo](#agentinfo) \| null | no   | Agent tree snapshot captured at invocation start   |
| `started_at`         | float                       | yes      | Unix timestamp when the invocation started        |
| `ended_at`           | float \| null               | no       | Unix timestamp when the invocation ended          |
| `duration_ms`        | float \| null               | no       | Wall-clock duration in milliseconds               |
| `total_input_tokens` | integer                     | yes      | Aggregated input token count (default 0)          |
| `total_output_tokens`| integer                     | yes      | Aggregated output token count (default 0)         |
| `llm_call_count`     | integer                     | yes      | Number of LLM calls in this invocation (default 0)|
| `tool_call_count`    | integer                     | yes      | Number of tool calls in this invocation (default 0)|
| `error_count`        | integer                     | yes      | Number of errors encountered (default 0)          |

---

## AgentSpanRecord

One `agent.run_async()` execution. Opened by `before_agent_callback`, closed by `after_agent_callback`.

```json
{
  "span_id": "string",
  "invocation_id": "string",
  "session_id": "string",
  "agent_name": "string",
  "agent_description": "string",
  "agent_type": "string",
  "branch": "root.planner",
  "started_at": 1234567890.123,
  "ended_at": 1234567890.456,
  "duration_ms": 333.0,
  "output": {},
  "error": "string"
}
```

| Field               | Type           | Required | Description                                              |
| ------------------- | -------------- | -------- | -------------------------------------------------------- |
| `span_id`           | string         | yes      | Unique span identifier                                   |
| `invocation_id`     | string         | yes      | Parent invocation identifier                             |
| `session_id`        | string         | yes      | Parent session identifier                                |
| `agent_name`        | string         | yes      | Agent name (e.g. `"planner"`)                            |
| `agent_description` | string \| null | no       | Human-readable description of the agent                  |
| `agent_type`        | string         | yes      | Python class name (`"LlmAgent"`, `"SequentialAgent"`, etc.) |
| `branch`            | string \| null | no       | Dot-separated agent hierarchy path (e.g. `"root.planner"`) |
| `started_at`        | float          | yes      | Unix timestamp when the agent span started               |
| `ended_at`          | float \| null  | no       | Unix timestamp when the agent span ended                 |
| `duration_ms`       | float \| null  | no       | Wall-clock duration in milliseconds                      |
| `output`            | object \| null | no       | Serialised `Content` if the after-callback produced output |
| `error`             | string \| null | no       | Error message if the agent span failed                   |

---

## LlmSpanRecord

One LLM `generate_content` call. Opened by `before_model_callback`, closed by `after_model_callback`.

```json
{
  "span_id": "string",
  "invocation_id": "string",
  "session_id": "string",
  "agent_name": "string",
  "request": {},
  "response": {},
  "started_at": 1234567890.123,
  "ended_at": 1234567890.456,
  "duration_ms": 333.0
}
```

| Field           | Type                                    | Required | Description                         |
| --------------- | --------------------------------------- | -------- | ----------------------------------- |
| `span_id`       | string                                  | yes      | Unique span identifier              |
| `invocation_id` | string                                  | yes      | Parent invocation identifier        |
| `session_id`    | string                                  | yes      | Parent session identifier           |
| `agent_name`    | string                                  | yes      | Name of the agent making the call   |
| `request`       | [LlmRequest](#llmrequest) \| null       | no       | Captured LLM request payload        |
| `response`      | [LlmResponseData](#llmresponsedata) \| null | no  | Captured LLM response payload       |
| `started_at`    | float                                   | yes      | Unix timestamp when the call started|
| `ended_at`      | float \| null                           | no       | Unix timestamp when the call ended  |
| `duration_ms`   | float \| null                           | no       | Wall-clock duration in milliseconds |

---

## ToolSpanRecord

One tool execution. Opened by `before_tool_callback`, closed by `after_tool_callback`.

```json
{
  "span_id": "string",
  "invocation_id": "string",
  "session_id": "string",
  "agent_name": "string",
  "tool_name": "string",
  "tool_description": "string",
  "tool_type": "FunctionTool",
  "is_long_running": false,
  "args": {},
  "result": {},
  "error": "string",
  "started_at": 1234567890.123,
  "ended_at": 1234567890.456,
  "duration_ms": 333.0
}
```

| Field              | Type           | Required | Description                                              |
| ------------------ | -------------- | -------- | -------------------------------------------------------- |
| `span_id`          | string         | yes      | Unique span identifier                                   |
| `invocation_id`    | string         | yes      | Parent invocation identifier                             |
| `session_id`       | string         | yes      | Parent session identifier                                |
| `agent_name`       | string         | yes      | Name of the agent that invoked the tool                  |
| `tool_name`        | string         | yes      | Tool name                                                |
| `tool_description` | string \| null | no       | Human-readable tool description                          |
| `tool_type`        | string         | yes      | Python class name (`"FunctionTool"`, `"McpTool"`, etc.)  |
| `is_long_running`  | boolean        | yes      | Whether the tool is long-running (default `false`)       |
| `args`             | object \| null | no       | Arguments passed to the tool                             |
| `result`           | object \| null | no       | Result returned by the tool                              |
| `error`            | string \| null | no       | Error message if the tool execution failed               |
| `started_at`       | float          | yes      | Unix timestamp when the tool execution started           |
| `ended_at`         | float \| null  | no       | Unix timestamp when the tool execution ended             |
| `duration_ms`      | float \| null  | no       | Wall-clock duration in milliseconds                      |

---

## EventRecord

Raw ADK `Event` passthrough. Created for every event yielded by the runner via `on_event_callback`.

```json
{
  "event_id": "string",
  "invocation_id": "string",
  "session_id": "string",
  "author": "string",
  "timestamp": 1234567890.123,
  "branch": "root.planner",
  "content": {},
  "partial": false,
  "is_final_response": false,
  "actions": {},
  "error_code": "string",
  "error_message": "string"
}
```

| Field               | Type                                          | Required | Description                                    |
| ------------------- | --------------------------------------------- | -------- | ---------------------------------------------- |
| `event_id`          | string                                        | yes      | Unique event identifier                        |
| `invocation_id`     | string                                        | yes      | Parent invocation identifier                   |
| `session_id`        | string                                        | yes      | Parent session identifier                      |
| `author`            | string                                        | yes      | `"user"` or the agent name                     |
| `timestamp`         | float                                         | yes      | Unix timestamp of the event                    |
| `branch`            | string \| null                                | no       | Dot-separated agent hierarchy path             |
| `content`           | object \| null                                | no       | Serialised `Content`                           |
| `partial`           | boolean \| null                               | no       | Whether this is a partial/streaming event      |
| `is_final_response` | boolean                                       | yes      | Whether this is the final response (default `false`) |
| `actions`           | [EventActionsData](#eventactionsdata) \| null | no       | Actions associated with this event             |
| `error_code`        | string \| null                                | no       | Error code if the event represents an error    |
| `error_message`     | string \| null                                | no       | Error message if the event represents an error |

---

## Shared Sub-Schemas

### AgentInfo

Metadata about an agent in the agent tree. Recursive — `sub_agents` contains nested `AgentInfo` objects.

```json
{
  "name": "string",
  "description": "string",
  "type": "LlmAgent",
  "sub_agents": []
}
```

| Field         | Type             | Required | Description                                           |
| ------------- | ---------------- | -------- | ----------------------------------------------------- |
| `name`        | string           | yes      | Agent name (e.g. `"planner"`)                         |
| `description` | string \| null   | no       | Human-readable description                            |
| `type`        | string           | yes      | Python class name (`"LlmAgent"`, `"SequentialAgent"`) |
| `sub_agents`  | AgentInfo[]      | yes      | Recursive children in the agent hierarchy (default `[]`) |

### LlmRequest

Captured LLM request payload, populated from `before_model_callback`.

```json
{
  "model": "string",
  "system_instruction": "string",
  "contents": [],
  "tools": [],
  "generation_config": {}
}
```

| Field                | Type                                            | Required | Description                               |
| -------------------- | ----------------------------------------------- | -------- | ----------------------------------------- |
| `model`              | string \| null                                  | no       | Model name (e.g. `"gemini-2.0-flash"`)    |
| `system_instruction` | string \| null                                  | no       | System prompt sent to the model           |
| `contents`           | object[] \| null                                | no       | Serialised chat history                   |
| `tools`              | [ToolInfo](#toolinfo)[] \| null                 | no       | Tool schemas available to the model       |
| `generation_config`  | [GenerationConfig](#generationconfig) \| null   | no       | Generation parameters                     |

### LlmResponseData

Captured LLM response payload, populated from `after_model_callback`.

```json
{
  "content": {},
  "finish_reason": "string",
  "model_version": "string",
  "error_code": "string",
  "error_message": "string",
  "usage": {}
}
```

| Field           | Type                                      | Required | Description                          |
| --------------- | ----------------------------------------- | -------- | ------------------------------------ |
| `content`       | object \| null                            | no       | Serialised `Content` from the model  |
| `finish_reason` | string \| null                            | no       | Why the model stopped generating     |
| `model_version` | string \| null                            | no       | Actual model version used            |
| `error_code`    | string \| null                            | no       | Error code if the call failed        |
| `error_message` | string \| null                            | no       | Error message if the call failed     |
| `usage`         | [UsageMetadata](#usagemetadata) \| null   | no       | Token usage counters                 |

### ToolInfo

Lightweight tool descriptor extracted from the LLM request.

```json
{
  "name": "string",
  "description": "string"
}
```

| Field         | Type           | Required | Description              |
| ------------- | -------------- | -------- | ------------------------ |
| `name`        | string         | yes      | Tool name                |
| `description` | string \| null | no       | Tool description         |

### GenerationConfig

Subset of `GenerateContentConfig` relevant for observability.

```json
{
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 1024,
  "stop_sequences": [],
  "candidate_count": 1
}
```

| Field              | Type             | Required | Description                     |
| ------------------ | ---------------- | -------- | ------------------------------- |
| `temperature`      | float \| null    | no       | Sampling temperature            |
| `top_p`            | float \| null    | no       | Top-p (nucleus) sampling        |
| `top_k`            | integer \| null  | no       | Top-k sampling                  |
| `max_output_tokens`| integer \| null  | no       | Maximum output tokens           |
| `stop_sequences`   | string[] \| null | no       | Stop sequences                  |
| `candidate_count`  | integer \| null  | no       | Number of candidates to generate|

### UsageMetadata

Token usage counters from the LLM response.

```json
{
  "input_tokens": 100,
  "output_tokens": 50,
  "total_tokens": 150,
  "cached_content_token_count": 0
}
```

| Field                        | Type            | Required | Description                        |
| ---------------------------- | --------------- | -------- | ---------------------------------- |
| `input_tokens`               | integer \| null | no       | Number of input tokens             |
| `output_tokens`              | integer \| null | no       | Number of output tokens            |
| `total_tokens`               | integer \| null | no       | Total token count                  |
| `cached_content_token_count` | integer \| null | no       | Tokens served from cache           |

### EventActionsData

Subset of `EventActions` relevant for observability.

```json
{
  "state_delta": {},
  "artifact_delta": {},
  "transfer_to_agent": "string",
  "escalate": false,
  "end_of_agent": false
}
```

| Field               | Type                      | Required | Description                                |
| ------------------- | ------------------------- | -------- | ------------------------------------------ |
| `state_delta`       | object \| null            | no       | State changes made by this event           |
| `artifact_delta`    | object (str→int) \| null  | no       | Artifact version changes                   |
| `transfer_to_agent` | string \| null            | no       | Agent to transfer control to               |
| `escalate`          | boolean \| null           | no       | Whether the agent is escalating            |
| `end_of_agent`      | boolean \| null           | no       | Whether this ends the current agent's turn |

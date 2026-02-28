# Sandbox Math Agent — Sandboxed Execution with Tinker

This example trains a math agent that runs inside a **sandboxed environment**
(local subprocess or Docker container) rather than inside the trainer process.
It is the sandbox counterpart of `adk_math`.

The agent code lives in a standalone `agent/` directory with its own
dependencies. During training the orchestrator uploads the agent code into
sandboxes, dispatches tasks over HTTP, and collects results through a
SQLite-backed result store. The agent only needs a standard OpenAI client —
no framework-specific imports required.

## Architecture

```
Host (trainer process)                          Sandbox (subprocess / container)
┌────────────────────────────────┐              ┌──────────────────────────────┐
│ AgentTrainer (Tinker backend)  │              │ worker_server.py             │
│  ├─ SdkWorkflowFactory        │              │  ├─ /health                  │
│  │   ├─ LiteLLM proxy         │◄── results ──│  ├─ /execute (fire & forget) │
│  │   │   ├─ TracingCallback    │              │  │   ├─ generate session_uid │
│  │   │   └─ ResultStore routes │              │  │   ├─ encode into proxy URL│
│  │   ├─ ExecutionResultStore   │              │  │   ├─ call rollout(task,   │
│  │   └─ SandboxOrchestrator   │── tasks ────►│  │   │       config)         │
│  │       └─ worker pool / sem  │              │  │   └─ POST result to proxy │
│  └─ UnifiedWorkflowEngine     │              │  └─ agent/                   │
│      └─ SdkWorkflow (per task) │              │      └─ agent.py            │
│          └─ orchestrator.exec()│              │          └─ rollout()        │
│                                │              │              └─ OpenAI client│
│ Tinker RL training loop        │              │                  ▲           │
└────────────────────────────────┘              └──────────────────│───────────┘
                                                                  │
                                LLM calls (proxied base_url) ─────┘
```

1. The trainer starts the LiteLLM proxy with `--enable-result-store`.
2. `SandboxOrchestrator` creates sandbox workers and uploads `agent/` + the
   built-in `worker_server.py`.
3. For each task, the orchestrator POSTs to a worker's `/execute` endpoint.
4. The worker generates a `session_uid`, encodes it into the proxy URL via
   a metadata slug, and calls `agent.rollout(task, config)`.
5. The agent makes standard OpenAI-compatible LLM calls through the proxied
   URL. The proxy's `TracingCallback` captures token IDs and logprobs.
6. The worker serialises the returned `list[Trajectory]` and POSTs the result
   back to the proxy's `/rllm/results/{execution_id}` route.
7. The trainer polls the `ExecutionResultStore`, deserialises trajectories,
   merges them with proxy traces, and builds training-ready Episodes.

## Files

| File | Description |
|------|-------------|
| `train.py` | Training script (host-side). Configures sandbox, launches trainer. Includes `--test-sandbox` smoke test. |
| `agent/agent.py` | Standalone agent: `rollout(task, config) -> list[Trajectory]`. Uses the OpenAI client against the proxied URL. |
| `agent/requirements.txt` | Agent-side dependencies (just `openai`). |

## Agent Contract

The agent module must expose a function with this signature:

```python
def rollout(task: dict, config: dict) -> list[Trajectory]:
    ...
```

The `config` dict always contains:

| Key | Description |
|-----|-------------|
| `base_url` | Proxied OpenAI-compatible endpoint with session metadata encoded in the URL |
| `session_uid` | Unique ID for trace correlation |
| `model_id` | Model name to use for completions |

The agent creates an OpenAI client pointing at `base_url`, runs its logic,
computes a reward, and returns one or more `Trajectory` objects.

## Quick Start

### 1. Prepare dataset

```bash
python -m examples.countdown.prepare_countdown_data
```

### 2. Smoke test (no GPU, no model)

Validates the full sandbox plumbing — sandbox creation, agent upload, task
dispatch, LLM call (mocked), result push, and result retrieval:

```bash
python -m examples.sdk.sandbox_math.train --test-sandbox
```

Expected output:

```
Agent directory: .../examples/sdk/sandbox_math/agent
Test server listening on http://127.0.0.1:18999/v1
Creating local sandbox orchestrator...
Dispatching test task...
Result: success=True, session_uid=...
  trajectories: 1 returned
  traj[0]: name=solver, reward=1.0, steps=1

Sandbox smoke test PASSED (reward=1.0)
```

### 3. Run training

```bash
python -m examples.sdk.sandbox_math.train \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-8B \
    training.group_size=8 \
    data.train_batch_size=4 \
    rllm.trainer.test_freq=5 \
    rllm.trainer.val_before_train=false
```

## Configuration

Sandbox settings live under `rllm.sdk.sandbox` and can be overridden on the
command line:

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `false` | Enable sandboxed execution |
| `backend` | `local` | Sandbox backend: `local`, `docker`, `modal`, `agentcore` |
| `agent_dir` | `""` | Path to the agent project directory |
| `agent_module` | `agent` | Python module name containing the rollout function |
| `agent_func` | `rollout` | Function name in the agent module |
| `pool_mode` | `persistent` | `persistent` (warm pool) or `per_task` (fresh sandbox per task) |
| `num_workers` | `8` | Number of persistent workers |
| `worker_port` | `8100` | Base port for worker servers |
| `image` | `python:3.11-slim` | Docker image (when `backend=docker`) |
| `install_rllm_sdk` | `true` | Install `rllm[sdk]` in the sandbox |
| `execution_timeout` | `600.0` | Max seconds to wait for a single task |

### Example: Docker backend

```bash
python -m examples.sdk.sandbox_math.train \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-8B \
    rllm.sdk.sandbox.backend=docker \
    rllm.sdk.sandbox.image=python:3.11-slim \
    rllm.sdk.sandbox.num_workers=16
```

### Example: Per-task sandboxes

```bash
python -m examples.sdk.sandbox_math.train \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-8B \
    rllm.sdk.sandbox.pool_mode=per_task \
    rllm.sdk.sandbox.max_concurrent=32
```

## How It Differs from Other Examples

| Aspect | `sandbox_math` (this) | `adk_math` | `strands_math` |
|--------|-----------------------|------------|----------------|
| Where agent runs | Sandbox (subprocess / Docker) | Trainer process | Trainer process |
| Agent framework | Plain OpenAI client | Google ADK | Strands Agents SDK |
| Agent provided as | Standalone directory (`agent/`) | `agent_run_func` callable | `agent_run_func` callable |
| Dependency isolation | Full (separate environment) | Shared with trainer | Shared with trainer |
| Result delivery | Fire-and-forget push to SQLite | Direct return value | Direct return value |
| Config entry point | `rllm.sdk.sandbox.*` | `agent_run_func=` kwarg | `agent_run_func=` kwarg |

# Strands Math Agent — SDK `agent_run_func` with Tinker

This example trains a Strands-based math agent using the SDK `agent_run_func`
pattern with the rLLM unified trainer and Tinker backend.

Unlike the `remote_strands_math` example (which runs the agent in a separate
container via `@rollout_endpoint`), this example runs the agent **locally**
inside the trainer process.  The unified trainer automatically wraps the agent
function in an `SdkWorkflow` adapter, handling trace collection, advantage
computation, and RL training.

## Architecture

```
Trainer Process
├── AgentTrainer (Tinker backend)
│   ├── SdkWorkflowFactory
│   │   ├── TinkerProxyManager → TinkerBackendServer
│   │   └── SqliteTraceStore (proxy TracingCallback writes here)
│   └── UnifiedWorkflowEngine
│       └── SdkWorkflow (per task)
│           └── strands_math_rollout()
│               ├── Strands Agent
│               │   └── OpenAIModel(base_url=RLLM_SDK_BASE_URL)
│               └── Returns scalar reward (float)
└── Tinker RL training loop (PPO/GRPO)
```

All LLM calls from the Strands agent flow through the trainer's proxy, which
captures token IDs and logprobs via `TracingCallback` → SQLite.  The agent
function only returns a scalar reward.  `SdkWorkflow` builds training-ready
Episodes from the proxy traces + reward automatically.

**Note:** `RLLMTrajectoryHookProvider` is NOT used here.  It is designed for
the `@rollout_endpoint` (remote) pattern where trajectory structure must be
returned over HTTP.  In the `agent_run_func` (local) pattern, the proxy
already captures everything needed.

## Files

| File | Description |
|------|-------------|
| `train.py` | Agent function + trainer: Strands math solver via `agent_run_func` (returns scalar reward) |

## Quick Start

### 1. Prepare dataset

```bash
python -m examples.countdown.prepare_countdown_data
```

### 2. Install Strands

```bash
pip install 'strands-agents[openai]'
```

### 3. Run the trainer

```bash
python -m examples.sdk.strands_math.train \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-8B \
    training.group_size=8 \
    validation.group_size=1 \
    training.learning_rate=2e-5 \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    rllm.trainer.test_freq=5 \
    rllm.trainer.val_before_train=true
```

## How It Differs from `remote_strands_math`

| Aspect | `strands_math` (this example) | `remote_strands_math` |
|--------|-------------------------------|----------------------|
| Agent location | Local (in trainer process) | Remote (HTTP endpoint) |
| Pattern | `agent_run_func` (returns float) | `@rollout_endpoint` (returns `list[Trajectory]`) |
| Tracing | Proxy `TracingCallback` (automatic) | `RLLMTrajectoryHookProvider` (explicit) |
| Proxy | Auto-managed by `SdkWorkflowFactory` | Trainer dispatches via HTTP |
| Deployment | Single process | Multi-container capable |
| Use case | Simple setup, fast iteration | Production, sandboxing |

## How It Differs from `unified_tinker`

| Aspect | `strands_math` (this example) | `unified_tinker` |
|--------|-------------------------------|-------------------|
| Agent framework | Strands SDK | Raw OpenAI client |
| LLM client | Strands `OpenAIModel` | `get_chat_client()` |
| Return type | `float` (scalar reward) | `float` (scalar reward) |
| Multi-turn | Supported (Strands agent loop) | Single LLM call |

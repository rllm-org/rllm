# ADK Math Agent — SDK `agent_run_func` with Tinker

This example trains a Google ADK-based math agent using the SDK `agent_run_func`
pattern with the rLLM unified trainer and Tinker backend.

Unlike the `adk_trajectory` example (which only collects traces), this example
runs the agent **locally** inside the trainer process and performs RL training.
The unified trainer automatically wraps the agent function in an `SdkWorkflow`
adapter, handling trace collection, advantage computation, and RL training.

## Architecture

```
Trainer Process
├── AgentTrainer (Tinker backend)
│   ├── SdkWorkflowFactory
│   │   ├── TinkerProxyManager → TinkerBackendServer
│   │   └── SqliteTraceStore (proxy TracingCallback writes here)
│   └── UnifiedWorkflowEngine
│       └── SdkWorkflow (per task)
│           └── adk_math_rollout()
│               ├── ADK Agent + LiteLLM model
│               │   └── LiteLlm(api_base=RLLM_SDK_BASE_URL)
│               └── Returns scalar reward (float)
└── Tinker RL training loop (PPO/GRPO)
```

All LLM calls from the ADK agent flow through the trainer's proxy via LiteLLM,
which captures token IDs and logprobs via `TracingCallback` → SQLite.  The agent
function returns a scalar reward.  `SdkWorkflow` builds training-ready Episodes
from the proxy traces + reward automatically.

## Files

| File | Description |
|------|-------------|
| `train.py` | Agent function + trainer: ADK math solver via `agent_run_func` (returns scalar reward) |

## Quick Start

### 1. Prepare dataset

```bash
python -m examples.countdown.prepare_countdown_data
```

### 2. Install Google ADK with LiteLLM

```bash
pip install 'google-adk[extensions]'
```

### 3. Run the trainer

```bash
python -m examples.sdk.adk_math.train \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-8B \
    training.group_size=8 \
    validation.group_size=1 \
    training.learning_rate=2e-5 \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    rllm.trainer.test_freq=5 \
    rllm.trainer.val_before_train=false
```

## How It Differs from `adk_trajectory`

| Aspect | `adk_math` (this example) | `adk_trajectory` |
|--------|---------------------------|------------------|
| Purpose | RL training | Trace collection only |
| Backend | Tinker (RL training loop) | None (standalone script) |
| Model | Local model via proxy | Gemini API (remote) |
| Output | Trained model checkpoints | Trajectory JSON |

## How It Differs from `strands_math`

| Aspect | `adk_math` (this example) | `strands_math` |
|--------|---------------------------|----------------|
| Agent framework | Google ADK | Strands Agents SDK |
| LLM client | ADK `LiteLlm` | Strands `OpenAIModel` |
| Runner | ADK `Runner` (async) | Strands `Agent.__call__` (sync) |
| Plugin | `RLLMTrajectoryPlugin` | `RLLMTrajectoryHookProvider` |
| Return type | `list[Trajectory]` | `list[Trajectory]` |

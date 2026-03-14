# Strands Agent

rLLM agent that runs benchmarks using the [Strands Agents](https://github.com/strands-agents/strands-agents) framework.

## Installation

```bash
pip install -e agenthub/strands_agent
```

## Usage

### CLI

```bash
rllm eval gsm8k --agent strands --model gpt-4o-mini --max-examples 5
rllm eval mmlu_pro --agent strands --model gpt-4o-mini --max-examples 5
```

### Python

```python
from rllm.experimental.eval.agent_loader import load_agent

agent = load_agent("strands")
episode = agent.run(task, config)
```

## How It Works

The agent implements the `AgentFlow` protocol using Strands' `Agent` and `OpenAIModel`:

1. Creates an `OpenAIModel` pointed at the rLLM eval proxy (`config.base_url`)
2. Attaches an `RLLMTrajectoryHookProvider` to capture LLM call traces via Strands' hook system
3. Adapts to any benchmark via `TaskSpec` — uses `spec.instruction` for the system prompt and `spec.render_input(task)` for user input
4. Bridges tools from `config.metadata["tools"]` (if any) to Strands tool functions
5. Runs the agent and returns an `Episode` with traced trajectories

## SDK Integration

The underlying `RLLMTrajectoryHookProvider` can also be used standalone:

```python
from strands import Agent
from strands.models.openai import OpenAIModel
from rllm.sdk.integrations.strands import RLLMTrajectoryHookProvider

hook_provider = RLLMTrajectoryHookProvider()
model = OpenAIModel(client_args={"api_key": "sk-..."}, model_id="gpt-4o")
agent = Agent(model=model, hooks=[hook_provider])

result = agent("What is 15 * 7 + 23?")

traj = hook_provider.get_trajectory()  # rLLM Trajectory with OpenAI-format steps
```

## Dependencies

- `rllm`
- `strands-agents[openai]`

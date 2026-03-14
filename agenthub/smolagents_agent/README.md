# SmolAgents Agent

rLLM agent that runs benchmarks using the [SmolAgents](https://github.com/huggingface/smolagents) framework.

## Installation

```bash
pip install -e agenthub/smolagents_agent
```

## Usage

### CLI

```bash
rllm eval gsm8k --agent smolagents --model gpt-4o-mini --max-examples 5
rllm eval mmlu_pro --agent smolagents --model gpt-4o-mini --max-examples 5
```

### Python

```python
from rllm.experimental.eval.agent_loader import load_agent

agent = load_agent("smolagents")
episode = agent.run(task, config)
```

## How It Works

The agent implements the `AgentFlow` protocol using SmolAgents' `ToolCallingAgent` and `OpenAIServerModel`:

1. Creates an `OpenAIServerModel` pointed at the rLLM eval proxy (`config.base_url`)
2. Wraps the model with `RLLMSmolAgentsTracer` to capture LLM call traces
3. Adapts to any benchmark via `TaskSpec` — uses `spec.instruction` for the system prompt and `spec.render_input(task)` for user input
4. Bridges tools from `config.metadata["tools"]` (if any) to SmolAgents `Tool` subclasses
5. Runs the agent and returns an `Episode` with traced trajectories

## SDK Integration

The underlying `RLLMSmolAgentsTracer` can also be used standalone:

```python
from smolagents import OpenAIServerModel, ToolCallingAgent
from rllm.sdk.integrations.smolagents import RLLMSmolAgentsTracer

tracer = RLLMSmolAgentsTracer()
model = OpenAIServerModel(model_id="gpt-4o")
wrapped = tracer.wrap_model(model)

agent = ToolCallingAgent(tools=[], model=wrapped, system_prompt="Solve the task.")
result = agent.run("What is 15 * 7 + 23?")

traj = tracer.get_trajectory()  # rLLM Trajectory with OpenAI-format steps
```

## Dependencies

- `rllm`
- `smolagents>=1.20.0`

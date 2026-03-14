# LangGraph Agent

rLLM agent that runs benchmarks using the [LangGraph](https://github.com/langchain-ai/langgraph) framework.

## Installation

```bash
pip install -e agenthub/langgraph_agent
```

## Usage

### CLI

```bash
rllm eval gsm8k --agent langgraph --model gpt-4o-mini --max-examples 5
rllm eval mmlu_pro --agent langgraph --model gpt-4o-mini --max-examples 5
```

### Python

```python
from rllm.experimental.eval.agent_loader import load_agent

agent = load_agent("langgraph")
episode = agent.run(task, config)
```

## How It Works

The agent implements the `AgentFlow` protocol using LangGraph's `StateGraph` and LangChain's `ChatOpenAI`:

1. Creates a `ChatOpenAI` model pointed at the rLLM eval proxy (`config.base_url`)
2. Attaches an `RLLMTrajectoryCallbackHandler` to capture LLM call traces via LangChain's callback system
3. Adapts to any benchmark via `TaskSpec` — uses `spec.instruction` for the system prompt and `spec.render_input(task)` for user input
4. Bridges tools from `config.metadata["tools"]` (if any) to LangChain `StructuredTool` instances
5. Builds a `StateGraph` with an agent node and (when tools are present) a tool execution node with conditional routing
6. Runs the graph and returns an `Episode` with traced trajectories

## SDK Integration

The underlying `RLLMTrajectoryCallbackHandler` can also be used standalone:

```python
from langchain_openai import ChatOpenAI
from rllm.sdk.integrations.langgraph import RLLMTrajectoryCallbackHandler

cb = RLLMTrajectoryCallbackHandler()
llm = ChatOpenAI(model="gpt-4o")

result = llm.invoke("What is 15 * 7 + 23?", config={"callbacks": [cb]})

traj = cb.get_trajectory()  # rLLM Trajectory with OpenAI-format steps
```

## Dependencies

- `rllm`
- `langgraph>=0.2.0`
- `langchain-openai>=0.2.0`

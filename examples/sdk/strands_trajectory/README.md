# Strands Agents + rLLM Trajectory Tracing

Collect rLLM-compatible training trajectories from Strands agents with minimal code changes.

## Quick Start

```python
from strands import Agent
from strands.models.openai import OpenAIModel
from rllm.sdk.integrations.strands import RLLMTrajectoryHookProvider

# Standard Strands agent (unchanged)
model = OpenAIModel(model_id="gpt-4o-mini")
hook_provider = RLLMTrajectoryHookProvider()
agent = Agent(model=model, system_prompt="...", hooks=[hook_provider])

# Run agent (unchanged)
result = agent("What is 15 * 7 + 23?")

# Get trajectory for training (1 line)
traj = hook_provider.get_trajectory()
traj.reward = my_reward_fn(traj.output, expected)
```

## Examples

| File | Description |
|------|-------------|
| `strands_agent_with_tracing.py` | Simple agent (no tools) with trajectory collection |
| `strands_tool_agent_with_tracing.py` | Tool-using agent with multi-step trajectory |
| `strands_multi_agent_with_tracing.py` | Multi-agent solver-judge pipeline with per-agent trajectories |

## How It Works

`RLLMTrajectoryHookProvider` is a Strands `HookProvider` that hooks into the agent execution lifecycle:

1. **BeforeInvocationEvent** -- resets state, captures the user input
2. **BeforeModelCallEvent** -- snapshots the current conversation messages and starts a timer
3. **AfterModelCallEvent** -- pairs the request messages with the model response, converts Bedrock types to OpenAI format, creates an rLLM `Trace`
4. **AfterToolCallEvent** -- annotates traces with tool execution metadata
5. **AfterInvocationEvent** -- assembles all traces into a `TrajectoryView`

Each LLM call becomes one `StepView` in the trajectory. The Bedrock-native message types (`Message` with `ContentBlock` containing `text`, `toolUse`, `toolResult`) are automatically converted to the OpenAI Chat Completions message format that rLLM's training pipelines expect.

## Multi-Agent: Per-Agent Trajectories

For a solver-then-judge pattern, pass the same `RLLMTrajectoryHookProvider` to
both agents.  Each agent invocation produces its own trajectory, and
`get_trajectories()` returns all of them.

```python
hook_provider = RLLMTrajectoryHookProvider()

solver = Agent(model=model, system_prompt="...", hooks=[hook_provider])
judge = Agent(model=model, system_prompt="...", hooks=[hook_provider])

# Run solver, then judge with the solver's output
solver_result = solver(problem)
solver_traj = hook_provider.get_trajectory()

judge_result = judge(f"Verify: {solver_result}")
judge_traj = hook_provider.get_trajectory()

# Each agent has its own trajectory
solver_traj.reward = solver_reward
judge_traj.reward = judge_reward
```

## What You Get

```
TrajectoryView
├── name: "solver"
├── steps: [StepView, StepView, ...]   # one per LLM call
│   ├── input:  {messages: [...]}       # OpenAI-format messages
│   ├── output: {message: {...}}        # OpenAI-format response
│   ├── reward: 0.0                     # set by your reward function
│   └── metadata: {tool_executions: [...]}
├── reward: 0.0                         # trajectory-level reward
├── input:  {message: "user query"}
└── output: "final agent response"
```

## Prerequisites

```bash
pip install 'strands-agents[openai]' rllm
export OPENAI_API_KEY=sk-...
```

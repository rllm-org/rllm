# OpenAI Agents SDK + rLLM Trajectory Tracing

Collect rLLM-compatible training trajectories from OpenAI Agents SDK agents with minimal code changes.

## Quick Start

```python
from agents import Agent, Runner
from rllm.sdk.integrations.openai_agents import RLLMTrajectoryHooks

# Standard Agents SDK agent (unchanged)
agent = Agent(name="solver", model="gpt-4o-mini", instructions="...")

# Add rLLM tracing (2 lines)
hooks = RLLMTrajectoryHooks()
result = Runner.run_sync(agent, "What is 15 * 7 + 23?", hooks=hooks)

# Get trajectory for training (1 line)
traj = hooks.get_trajectory()
traj.reward = my_reward_fn(traj.output, expected)
```

## Examples

| File | Description |
|------|-------------|
| `openai_agent_with_tracing.py` | Simple agent (no tools) with trajectory collection |
| `openai_tool_agent_with_tracing.py` | Tool-using agent with multi-step trajectory |
| `openai_multi_agent_with_tracing.py` | Multi-agent solver-judge (agents as tools) with per-agent trajectories |

## How It Works

`RLLMTrajectoryHooks` is an OpenAI Agents SDK `RunHooks` subclass that hooks into the agent execution lifecycle:

1. **on_llm_start** -- stores the outgoing request (system prompt + input items)
2. **on_llm_end** -- pairs request with `ModelResponse`, converts Responses API types to Chat Completions format, creates an rLLM `Trace`
3. **on_tool_end** -- annotates traces with tool execution metadata
4. **on_agent_end** -- assembles all traces into a `TrajectoryView`

Each LLM call becomes one `StepView` in the trajectory. The Responses API types (`ResponseOutputMessage`, `ResponseFunctionToolCall`, etc.) are automatically converted to the Chat Completions message format that rLLM's training pipelines expect.

## Multi-Agent: Per-Agent Trajectories

In multi-agent systems using the "agents as tools" pattern, each sub-agent that makes LLM calls gets its own trajectory. Pass the same `hooks` instance to both `Runner.run()` and each `as_tool()` call so all LLM calls are captured:

```python
from agents import Agent, Runner
from rllm.sdk.integrations.openai_agents import RLLMTrajectoryHooks

solver = Agent(name="solver", model="gpt-4o-mini", instructions="...")
judge = Agent(name="judge", model="gpt-4o-mini", instructions="...")

hooks = RLLMTrajectoryHooks()

orchestrator = Agent(
    name="orchestrator", model="gpt-4o-mini",
    instructions="...",
    tools=[
        solver.as_tool(tool_name="solve", tool_description="...", hooks=hooks),
        judge.as_tool(tool_name="verify", tool_description="...", hooks=hooks),
    ],
)

result = await Runner.run(orchestrator, "Solve this problem...", hooks=hooks)

# After running...
per_agent = hooks.get_trajectories_by_agent()
# => {"orchestrator": TrajectoryView, "solver": TrajectoryView, "judge": TrajectoryView}

per_agent["solver"].reward = solver_reward
per_agent["judge"].reward = judge_reward
```

> **Note:** Use "agents as tools" (not handoffs) for orchestrator patterns.
> Handoffs transfer control one-way, so the orchestrator cannot call a second
> sub-agent after the first one finishes.

## What You Get

```
TrajectoryView
├── name: "calculator_agent"
├── steps: [StepView, StepView, ...]   # one per LLM call
│   ├── input:  {messages: [...]}       # Chat Completions-format messages
│   ├── output: {message: {...}}        # Chat Completions-format response
│   ├── reward: 0.0                     # set by your reward function
│   └── metadata: {tool_executions: [...]}
├── reward: 0.0                         # trajectory-level reward
├── input:  {message: "user query"}
└── output: "final agent response"
```

## Prerequisites

```bash
pip install openai-agents rllm
```

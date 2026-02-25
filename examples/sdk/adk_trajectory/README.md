# ADK + rLLM Trajectory Tracing

Collect rLLM-compatible training trajectories from Google ADK agents with minimal code changes.

## Quick Start

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from rllm.sdk.integrations.adk import RLLMTrajectoryPlugin

# Standard ADK agent (unchanged)
agent = Agent(name="solver", model="gemini-2.5-flash", instruction="...")

# Add rLLM tracing (2 lines)
plugin = RLLMTrajectoryPlugin()
runner = Runner(
    app_name="my_app",
    agent=agent,
    session_service=InMemorySessionService(),
    plugins=[plugin],
)

# Run agent (unchanged)
for event in runner.run(user_id="u1", session_id="s1", new_message=msg):
    pass

# Get trajectory for training (1 line)
traj = plugin.get_trajectory()
traj.reward = my_reward_fn(traj.output, expected)
```

## Examples

| File | Description |
|------|-------------|
| `adk_agent_with_tracing.py` | Simple agent (no tools) with trajectory collection |
| `adk_tool_agent_with_tracing.py` | Tool-using agent with multi-step trajectory |
| `adk_multi_agent_with_tracing.py` | Multi-agent solver-judge pipeline with per-agent trajectories |

## How It Works

`RLLMTrajectoryPlugin` is a Google ADK `BasePlugin` that hooks into the agent execution lifecycle:

1. **before_model_callback** -- stores the outgoing `LlmRequest`
2. **after_model_callback** -- pairs request with response, converts Gemini types to OpenAI format, creates an rLLM `Trace`
3. **after_tool_callback** -- annotates traces with tool execution metadata
4. **after_run_callback** -- assembles all traces into a `TrajectoryView`

Each LLM call becomes one `StepView` in the trajectory. The Gemini-native types (`types.Content`, `LlmRequest`, `LlmResponse`) are automatically converted to the OpenAI-compatible message format that rLLM's training pipelines expect.

## Multi-Agent: Per-Agent Trajectories

For a solver-then-judge pattern, run each agent as a separate invocation through
the same plugin. Each invocation produces its own trajectory, and
`get_trajectories()` returns all of them.

```python
plugin = RLLMTrajectoryPlugin()

solver_runner = Runner(agent=solver, plugins=[plugin], ...)
judge_runner  = Runner(agent=judge,  plugins=[plugin], ...)

# Run solver, then judge with the solver's output
for event in solver_runner.run_async(..., new_message=problem):
    ...
solver_traj = plugin.get_trajectory()

for event in judge_runner.run_async(..., new_message=judge_input):
    ...
judge_traj = plugin.get_trajectory()

# Each agent has its own trajectory
solver_traj.reward = solver_reward
judge_traj.reward  = judge_reward
```

> **Note:** ADK's `sub_agents` are a delegation mechanism -- the parent
> transfers control and the run ends when the sub-agent finishes. For
> sequential multi-agent flows (solver then judge), use separate runner
> invocations instead.

## What You Get

```
TrajectoryView
├── name: "calculator_agent"
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
pip install google-adk rllm
```

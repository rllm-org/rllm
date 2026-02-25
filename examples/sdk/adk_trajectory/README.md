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
| `adk_multi_agent_with_tracing.py` | Multi-agent solver-judge with per-agent trajectories |

## How It Works

`RLLMTrajectoryPlugin` is a Google ADK `BasePlugin` that hooks into the agent execution lifecycle:

1. **before_model_callback** -- stores the outgoing `LlmRequest`
2. **after_model_callback** -- pairs request with response, converts Gemini types to OpenAI format, creates an rLLM `Trace`
3. **after_tool_callback** -- annotates traces with tool execution metadata
4. **after_run_callback** -- assembles all traces into a `TrajectoryView`

Each LLM call becomes one `StepView` in the trajectory. The Gemini-native types (`types.Content`, `LlmRequest`, `LlmResponse`) are automatically converted to the OpenAI-compatible message format that rLLM's training pipelines expect.

## Multi-Agent: Per-Agent Trajectories

In multi-agent systems, each sub-agent that makes LLM calls gets its own trajectory:

```python
coordinator = LlmAgent(
    name="coordinator", model="gemini-2.5-flash",
    sub_agents=[solver, judge],
)

plugin = RLLMTrajectoryPlugin()
runner = Runner(..., plugins=[plugin])

# After running...
per_agent = plugin.get_trajectories_by_agent()
# => {"coordinator": TrajectoryView, "solver": TrajectoryView, "judge": TrajectoryView}

per_agent["solver"].reward = solver_reward
per_agent["judge"].reward = judge_reward
```

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

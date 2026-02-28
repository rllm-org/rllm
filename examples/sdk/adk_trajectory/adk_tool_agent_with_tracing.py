"""Example: ADK tool-using agent with rLLM trajectory tracing.

Demonstrates tracing an ADK agent that uses tools.  The plugin captures
every LLM call (including the tool-calling rounds) and all tool execution
metadata, producing a multi-step ``Trajectory``.

Prerequisites:
    pip install google-adk rllm
    export GOOGLE_API_KEY=<your-key>   # or configure Vertex AI credentials

Usage:
    python adk_tool_agent_with_tracing.py
"""

import asyncio
import math
import os
import sys

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from rllm.sdk.integrations.adk import RLLMTrajectoryPlugin

# ---------------------------------------------------------------------------
# Tools (standard ADK function tools)
# ---------------------------------------------------------------------------


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A Python math expression to evaluate (e.g. "2 + 3 * 4").

    Returns:
        The numeric result as a string.
    """
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "int": int, "float": float})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

agent = Agent(
    name="calculator_agent",
    model="gemini-2.5-flash",
    instruction=("You are a precise math assistant. Use the calculator tool for all computations. Show your reasoning, then give the final answer."),
    tools=[calculator],
)

# ---------------------------------------------------------------------------
# rLLM tracing
# ---------------------------------------------------------------------------

trajectory_plugin = RLLMTrajectoryPlugin()

runner = Runner(
    app_name="calc_agent",
    agent=agent,
    session_service=InMemorySessionService(),
    plugins=[trajectory_plugin],
    auto_create_session=True,
)


# ---------------------------------------------------------------------------
# Run and inspect
# ---------------------------------------------------------------------------


async def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("ERROR: GOOGLE_API_KEY not set.\n  export GOOGLE_API_KEY=<your-key>\nGet one at https://aistudio.google.com/apikey")

    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="A rectangle has width 17.5 and height 9.3.  What is its area and the length of its diagonal?")],
    )

    print("=== Running tool-using ADK agent with rLLM tracing ===\n")

    async for event in runner.run_async(
        user_id="user_1",
        session_id="session_tool",
        new_message=user_message,
    ):
        if event.content and event.content.parts:
            text = "".join(p.text for p in event.content.parts if p.text)
            if text:
                print(f"[{event.author}] {text}")

    traj = trajectory_plugin.get_trajectory()

    print("\n=== Trajectory Summary ===")
    print(f"Agent:          {traj.name}")
    print(f"LLM calls:      {len(traj.steps)} (includes tool-calling rounds)")
    print(f"Model:          {traj.metadata.get('model', 'unknown')}")

    for i, step in enumerate(traj.steps):
        print(f"\n--- Step {i + 1} ---")
        output = step.output or {}
        msg = output.get("message", {}) if isinstance(output, dict) else {}
        has_tool_calls = "tool_calls" in msg if isinstance(msg, dict) else False
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        print(f"  Has tool_calls: {has_tool_calls}")
        if content:
            print(f"  Content:        {str(content)[:80]}...")
        tool_execs = (step.metadata or {}).get("tool_executions", [])
        if tool_execs:
            for te in tool_execs:
                print(f"  Tool executed:  {te['tool_name']}({te['tool_args']})")

    # Assign reward based on expected values
    expected_area = 17.5 * 9.3  # 162.75
    if traj.output and str(expected_area) in traj.output:
        traj.reward = 1.0
        print(f"\nReward: {traj.reward} (area correct)")
    else:
        traj.reward = 0.0
        print(f"\nReward: {traj.reward}")


if __name__ == "__main__":
    asyncio.run(main())

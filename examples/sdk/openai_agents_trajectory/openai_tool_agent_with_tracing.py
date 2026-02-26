"""Example: OpenAI Agents SDK tool-using agent with rLLM trajectory tracing.

Demonstrates tracing an agent that uses function tools.  The hooks capture
every LLM call (including the tool-calling rounds) and all tool execution
metadata, producing a multi-step ``Trajectory``.

Prerequisites:
    pip install openai-agents rllm
    export OPENAI_API_KEY=<your-key>

Usage:
    python openai_tool_agent_with_tracing.py
"""

import asyncio
import math
import os
import sys

from agents import Agent, Runner, function_tool

from rllm.sdk.integrations.openai_agents import RLLMTrajectoryHooks

# ---------------------------------------------------------------------------
# Tools (standard Agents SDK function tools)
# ---------------------------------------------------------------------------


@function_tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A Python math expression to evaluate (e.g. "2 + 3 * 4").
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
    model="gpt-4o-mini",
    instructions=("You are a precise math assistant. Use the calculator tool for all computations. Show your reasoning, then give the final answer."),
    tools=[calculator],
)


# ---------------------------------------------------------------------------
# Run and inspect
# ---------------------------------------------------------------------------


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set.\n  export OPENAI_API_KEY=sk-...\nGet one at https://platform.openai.com/api-keys")

    print("=== Running tool-using OpenAI agent with rLLM tracing ===\n")

    hooks = RLLMTrajectoryHooks()
    result = await Runner.run(
        agent,
        "A rectangle has width 17.5 and height 9.3. What is its area and the length of its diagonal?",
        hooks=hooks,
    )

    print(f"Agent output:\n{result.final_output}\n")

    traj = hooks.get_trajectory()

    print("=== Trajectory Summary ===")
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
                print(f"  Tool executed:  {te['tool_name']}({te.get('tool_result', '')[:60]})")

    # Assign reward based on expected values
    expected_area = 17.5 * 9.3  # 162.75
    if traj.output and str(expected_area) in str(traj.output):
        traj.reward = 1.0
        print(f"\nReward: {traj.reward} (area correct)")
    else:
        traj.reward = 0.0
        print(f"\nReward: {traj.reward}")


if __name__ == "__main__":
    asyncio.run(main())

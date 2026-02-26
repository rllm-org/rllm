"""Example: Strands tool-using agent with rLLM trajectory tracing.

Demonstrates tracing a Strands agent that uses tools.  The hook provider
captures every LLM call (including the tool-calling rounds) and all tool
execution metadata, producing a multi-step ``TrajectoryView``.

Prerequisites:
    pip install 'strands-agents[openai]' rllm
    export OPENAI_API_KEY=<your-key>

Usage:
    python strands_tool_agent_with_tracing.py
"""

import math
import os
import sys

from strands import Agent, tool
from strands.models.openai import OpenAIModel

from rllm.sdk.integrations.strands import RLLMTrajectoryHookProvider

# ---------------------------------------------------------------------------
# Tools (standard Strands @tool functions)
# ---------------------------------------------------------------------------


@tool
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

model = OpenAIModel(
    client_args={"api_key": os.environ.get("OPENAI_API_KEY", "")},
    model_id="gpt-4o-mini",
)

hook_provider = RLLMTrajectoryHookProvider()

agent = Agent(
    model=model,
    system_prompt="You are a precise math assistant. Use the calculator tool for all computations. Show your reasoning, then give the final answer.",
    tools=[calculator],
    hooks=[hook_provider],
    callback_handler=None,
)


# ---------------------------------------------------------------------------
# Run and inspect
# ---------------------------------------------------------------------------


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set.\n  export OPENAI_API_KEY=sk-...\nGet one at https://platform.openai.com/api-keys")

    print("=== Running tool-using Strands agent with rLLM tracing ===\n")

    result = agent("A rectangle has width 17.5 and height 9.3. What is its area and the length of its diagonal?")
    print(f"\nAgent output:\n{result}\n")

    traj = hook_provider.get_trajectory()

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
    main()

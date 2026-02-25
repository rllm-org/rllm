"""Example: OpenAI Agents SDK agent with rLLM trajectory tracing.

Shows how to add rLLM trajectory collection to a standard OpenAI Agents SDK
agent with just a few extra lines.  The collected ``TrajectoryView`` can be
used for SFT distillation or RL training.

Prerequisites:
    pip install openai-agents rllm
    export OPENAI_API_KEY=<your-key>

Usage:
    python openai_agent_with_tracing.py
"""

import asyncio
import os
import sys

from agents import Agent, Runner

from rllm.sdk.integrations.openai_agents import RLLMTrajectoryHooks

# ---------------------------------------------------------------------------
# 1. Define your agent (standard Agents SDK code -- nothing rLLM-specific)
# ---------------------------------------------------------------------------

agent = Agent(
    name="math_solver",
    model="gpt-4o-mini",
    instructions=("You are a math tutor. Solve the given problem step by step, then give the final numeric answer on the last line."),
)


# ---------------------------------------------------------------------------
# 2. Run the agent with rLLM tracing hooks
# ---------------------------------------------------------------------------


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set.\n  export OPENAI_API_KEY=sk-...\nGet one at https://platform.openai.com/api-keys")

    print("=== Running OpenAI agent with rLLM tracing ===\n")

    # Create hooks and pass to Runner -- this is all you need for tracing
    hooks = RLLMTrajectoryHooks()
    result = await Runner.run(
        agent,
        "What is 15 * 7 + 23?",
        hooks=hooks,
    )

    print(f"Agent output: {result.final_output}\n")

    # -------------------------------------------------------------------
    # 3. Get the trajectory -- ready for reward assignment & training
    # -------------------------------------------------------------------

    traj = hooks.get_trajectory()

    print("=== Trajectory Summary ===")
    print(f"Agent:          {traj.name}")
    print(f"LLM calls:      {len(traj.steps)}")
    print(f"User input:     {traj.input}")
    print(f"Final output:   {str(traj.output)[:100]}...")

    for i, step in enumerate(traj.steps):
        print(f"\n--- Step {i + 1} ---")
        print(f"  Trace ID:  {step.id}")
        n_msgs = len(step.input.get("messages", [])) if isinstance(step.input, dict) else 0
        print(f"  Messages:  {n_msgs}")
        print(f"  Reward:    {step.reward}")

    # -------------------------------------------------------------------
    # 4. Assign rewards (example: simple correctness check)
    # -------------------------------------------------------------------

    expected_answer = "128"
    if traj.output and expected_answer in str(traj.output):
        traj.reward = 1.0
        for step in traj.steps:
            step.reward = 1.0
        print(f"\nReward: {traj.reward} (correct!)")
    else:
        traj.reward = 0.0
        print(f"\nReward: {traj.reward} (incorrect)")

    # The TrajectoryView is now ready for:
    #   - SFT distillation (use traj.steps[i].input/output as training pairs)
    #   - RL training with rLLM (pass to rLLM trainer)
    print(f"\nTrajectoryView object: {traj.model_dump_json(indent=2)[:500]}...")


if __name__ == "__main__":
    asyncio.run(main())

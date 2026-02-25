"""Example: ADK multi-agent system with per-agent rLLM trajectory tracing.

Demonstrates a solver-judge pattern built with ADK's multi-agent delegation.
The ``RLLMTrajectoryPlugin`` automatically groups LLM calls by sub-agent so
each agent gets its own ``TrajectoryView`` -- enabling per-agent reward
assignment for RL training.

Prerequisites:
    pip install google-adk rllm
    export GOOGLE_API_KEY=...   # or configure a Gemini model

Usage:
    python adk_multi_agent_with_tracing.py
"""

import asyncio
import os
import sys

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from rllm.sdk.integrations.adk import RLLMTrajectoryPlugin

# ---------------------------------------------------------------------------
# 1. Define sub-agents
# ---------------------------------------------------------------------------

solver = LlmAgent(
    name="solver",
    model="gemini-2.5-flash",
    instruction=("You are a math solver. Given a math problem, work through it step by step and provide the final numeric answer on the last line in the format: ANSWER: <number>"),
    description="Solves math problems step by step.",
)

judge = LlmAgent(
    name="judge",
    model="gemini-2.5-flash",
    instruction=("You are a math verifier. You will receive a math problem and a proposed solution. Check if the solution is correct. Respond with CORRECT if the answer is right, or INCORRECT with the right answer if it is wrong."),
    description="Verifies whether a proposed math solution is correct.",
)

# ---------------------------------------------------------------------------
# 2. Define the coordinator that delegates to sub-agents
# ---------------------------------------------------------------------------

coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction=("You coordinate solving and verifying math problems.\n1. First, delegate the problem to the 'solver' agent to get a solution.\n2. Then, delegate to the 'judge' agent to verify the solution.\n3. Finally, report the verified answer to the user."),
    description="Coordinates the solver and judge agents.",
    sub_agents=[solver, judge],
)

# ---------------------------------------------------------------------------
# 3. Set up runner with rLLM tracing
# ---------------------------------------------------------------------------

trajectory_plugin = RLLMTrajectoryPlugin()

runner = Runner(
    app_name="solver_judge",
    agent=coordinator,
    session_service=InMemorySessionService(),
    plugins=[trajectory_plugin],
    auto_create_session=True,
)


# ---------------------------------------------------------------------------
# 4. Run and inspect per-agent trajectories
# ---------------------------------------------------------------------------


async def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("ERROR: GOOGLE_API_KEY not set.\n  export GOOGLE_API_KEY=<your-key>\nGet one at https://aistudio.google.com/apikey")

    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="What is the sum of the first 20 prime numbers?")],
    )

    print("=== Running multi-agent system with rLLM tracing ===\n")

    async for event in runner.run_async(
        user_id="user_1",
        session_id="session_multi",
        new_message=user_message,
    ):
        if event.content and event.content.parts:
            text = "".join(p.text for p in event.content.parts if p.text)
            if text:
                print(f"[{event.author}] {text[:200]}")

    # -------------------------------------------------------------------
    # 5. Get the COMBINED trajectory (all agents merged)
    # -------------------------------------------------------------------

    combined = trajectory_plugin.get_trajectory()
    print(f"\n{'=' * 60}")
    print("COMBINED TRAJECTORY")
    print(f"{'=' * 60}")
    print(f"  Total LLM calls: {len(combined.steps)}")
    print(f"  Agents involved:  {combined.metadata.get('num_llm_calls')}")

    # -------------------------------------------------------------------
    # 6. Get PER-AGENT trajectories for independent reward assignment
    # -------------------------------------------------------------------

    per_agent = trajectory_plugin.get_trajectories_by_agent()

    for agent_name, traj in per_agent.items():
        print(f"\n{'=' * 60}")
        print(f"TRAJECTORY: {agent_name}")
        print(f"{'=' * 60}")
        print(f"  LLM calls: {len(traj.steps)}")
        print(f"  Output:    {str(traj.output)[:120] if traj.output else '(none)'}...")

        for i, step in enumerate(traj.steps):
            output = step.output or {}
            msg = output.get("message", {}) if isinstance(output, dict) else {}
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            has_tools = "tool_calls" in msg if isinstance(msg, dict) else False
            print(f"  Step {i + 1}: {'[tool_call]' if has_tools else content[:80]}...")

    # -------------------------------------------------------------------
    # 7. Assign rewards per agent
    # -------------------------------------------------------------------

    expected = "639"  # sum of first 20 primes: 2+3+5+7+11+13+17+19+23+29+31+37+41+43+47+53+59+61+67+71

    if "solver" in per_agent:
        solver_traj = per_agent["solver"]
        if solver_traj.output and expected in str(solver_traj.output):
            solver_traj.reward = 1.0
        else:
            solver_traj.reward = 0.0
        print(f"\nSolver reward:  {solver_traj.reward}")

    if "judge" in per_agent:
        judge_traj = per_agent["judge"]
        if judge_traj.output and "CORRECT" in str(judge_traj.output).upper():
            judge_traj.reward = 1.0
        else:
            judge_traj.reward = 0.0
        print(f"Judge reward:   {judge_traj.reward}")

    # The per-agent TrajectoryViews are now ready for:
    #   - SFT distillation of individual agent behaviors
    #   - Per-agent RL reward shaping
    print(f"\nCollected {len(per_agent)} per-agent trajectories for training.")


if __name__ == "__main__":
    asyncio.run(main())

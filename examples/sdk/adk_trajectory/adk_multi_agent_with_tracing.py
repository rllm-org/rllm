"""Example: ADK multi-agent system with per-agent rLLM trajectory tracing.

Demonstrates a solver-judge pipeline where each agent runs independently and
gets its own ``Trajectory`` -- enabling per-agent reward assignment for
RL training.

Note:
    ADK's ``sub_agents`` are a delegation mechanism (like handoffs): the
    parent transfers control and the run ends when the sub-agent finishes.
    The parent never gets a second turn.  For a solver-then-judge pattern,
    we run each agent as a separate invocation through the same plugin so
    both trajectories are captured.

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
# 1. Define agents
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
# 2. Set up runners with the SAME rLLM plugin for both agents.
#    Each runner.run_async() call produces one trajectory; the plugin
#    accumulates all of them via get_trajectories().
# ---------------------------------------------------------------------------

trajectory_plugin = RLLMTrajectoryPlugin()
session_service = InMemorySessionService()

solver_runner = Runner(
    app_name="solver_judge",
    agent=solver,
    session_service=session_service,
    plugins=[trajectory_plugin],
    auto_create_session=True,
)

judge_runner = Runner(
    app_name="solver_judge",
    agent=judge,
    session_service=session_service,
    plugins=[trajectory_plugin],
    auto_create_session=True,
)


# ---------------------------------------------------------------------------
# 3. Run the pipeline: solver -> judge
# ---------------------------------------------------------------------------


async def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("ERROR: GOOGLE_API_KEY not set.\n  export GOOGLE_API_KEY=<your-key>\nGet one at https://aistudio.google.com/apikey")

    problem = "What is the sum of the first 20 prime numbers?"

    # --- Step 1: Run the solver ---
    print("=== Running solver agent ===\n")

    solver_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=problem)],
    )

    solver_output = ""
    async for event in solver_runner.run_async(
        user_id="user_1",
        session_id="session_solver",
        new_message=solver_message,
    ):
        if event.content and event.content.parts:
            text = "".join(p.text for p in event.content.parts if p.text)
            if text:
                solver_output = text
                print(f"[solver] {text[:200]}")

    solver_traj = trajectory_plugin.get_trajectory()
    print(f"\nSolver trajectory: {len(solver_traj.steps)} LLM call(s)")

    # --- Step 2: Run the judge with the solver's output ---
    print("\n=== Running judge agent ===\n")

    judge_input = f"Problem: {problem}\n\nProposed solution:\n{solver_output}\n\nIs this solution correct?"
    judge_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=judge_input)],
    )

    judge_output = ""
    async for event in judge_runner.run_async(
        user_id="user_1",
        session_id="session_judge",
        new_message=judge_message,
    ):
        if event.content and event.content.parts:
            text = "".join(p.text for p in event.content.parts if p.text)
            if text:
                judge_output = text
                print(f"[judge] {judge_output[:200]}")

    judge_traj = trajectory_plugin.get_trajectory()
    print(f"\nJudge trajectory: {len(judge_traj.steps)} LLM call(s)")

    # -------------------------------------------------------------------
    # 4. Inspect all trajectories (one per agent run)
    # -------------------------------------------------------------------

    all_trajs = trajectory_plugin.get_trajectories()

    print(f"\n{'=' * 60}")
    print(f"COLLECTED {len(all_trajs)} TRAJECTORIES")
    print(f"{'=' * 60}")

    for idx, traj in enumerate(all_trajs):
        print(f"\n--- Trajectory {idx + 1}: {traj.name} ---")
        print(f"  LLM calls: {len(traj.steps)}")
        print(f"  Output:    {str(traj.output)[:120] if traj.output else '(none)'}...")

        for i, step in enumerate(traj.steps):
            output = step.output or {}
            msg = output.get("message", {}) if isinstance(output, dict) else {}
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            has_tools = "tool_calls" in msg if isinstance(msg, dict) else False
            print(f"  Step {i + 1}: {'[tool_call]' if has_tools else content[:80]}...")

    # -------------------------------------------------------------------
    # 5. Assign rewards per agent
    # -------------------------------------------------------------------

    expected = "639"  # sum of first 20 primes

    if solver_traj.output and expected in str(solver_traj.output):
        solver_traj.reward = 1.0
    else:
        solver_traj.reward = 0.0
    print(f"\nSolver reward:  {solver_traj.reward}")

    if judge_traj.output and "CORRECT" in str(judge_traj.output).upper():
        judge_traj.reward = 1.0
    else:
        judge_traj.reward = 0.0
    print(f"Judge reward:   {judge_traj.reward}")

    print(f"\nCollected {len(all_trajs)} trajectories for training.")


if __name__ == "__main__":
    asyncio.run(main())

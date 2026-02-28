"""Example: Strands multi-agent system with per-agent rLLM trajectory tracing.

Demonstrates a solver-judge pipeline where each agent runs independently and
gets its own ``Trajectory`` -- enabling per-agent reward assignment for
RL training.

The same ``RLLMTrajectoryHookProvider`` is passed to both agents so that
``get_trajectories()`` returns all collected trajectories.

Prerequisites:
    pip install 'strands-agents[openai]' rllm
    export OPENAI_API_KEY=<your-key>

Usage:
    python strands_multi_agent_with_tracing.py
"""

import os
import sys

from strands import Agent
from strands.models.openai import OpenAIModel

from rllm.sdk.integrations.strands import RLLMTrajectoryHookProvider

# ---------------------------------------------------------------------------
# 1. Define agents
# ---------------------------------------------------------------------------

model = OpenAIModel(
    client_args={"api_key": os.environ.get("OPENAI_API_KEY", "")},
    model_id="gpt-4o-mini",
)

hook_provider = RLLMTrajectoryHookProvider()

solver = Agent(
    name="solver",
    model=model,
    system_prompt=("You are a math solver. Given a math problem, work through it step by step and provide the final numeric answer on the last line in the format: ANSWER: <number>"),
    hooks=[hook_provider],
    callback_handler=None,
)

judge = Agent(
    name="judge",
    model=model,
    system_prompt=("You are a math verifier. You will receive a math problem and a proposed solution. Check if the solution is correct. Respond with CORRECT if the answer is right, or INCORRECT with the right answer if it is wrong."),
    hooks=[hook_provider],
    callback_handler=None,
)


# ---------------------------------------------------------------------------
# 2. Run the pipeline: solver -> judge
# ---------------------------------------------------------------------------


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set.\n  export OPENAI_API_KEY=sk-...\nGet one at https://platform.openai.com/api-keys")

    problem = "What is the sum of the first 20 prime numbers?"

    # --- Step 1: Run the solver ---
    print("=== Running solver agent ===\n")
    solver_result = solver(problem)
    solver_output = str(solver_result)
    print(f"[solver] {solver_output[:200]}")

    solver_traj = hook_provider.get_trajectory()
    print(f"\nSolver trajectory: {len(solver_traj.steps)} LLM call(s)")

    # --- Step 2: Run the judge with the solver's output ---
    print("\n=== Running judge agent ===\n")
    judge_input = f"Problem: {problem}\n\nProposed solution:\n{solver_output}\n\nIs this solution correct?"
    judge_result = judge(judge_input)
    judge_output = str(judge_result)
    print(f"[judge] {judge_output[:200]}")

    judge_traj = hook_provider.get_trajectory()
    print(f"\nJudge trajectory: {len(judge_traj.steps)} LLM call(s)")

    # -------------------------------------------------------------------
    # 3. Inspect all trajectories (one per agent run)
    # -------------------------------------------------------------------

    all_trajs = hook_provider.get_trajectories()

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
            print(f"  Step {i + 1}: {'[tool_call]' if has_tools else str(content)[:80]}...")

    # -------------------------------------------------------------------
    # 4. Assign rewards per agent
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
    main()

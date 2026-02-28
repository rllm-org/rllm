"""Example: OpenAI Agents SDK multi-agent system with per-agent rLLM trajectory tracing.

Demonstrates a solver-judge pattern using the "agents as tools" mechanism.
The orchestrator agent calls solver and judge as tools, keeping control of the
conversation. ``RLLMTrajectoryHooks`` automatically groups LLM calls by
sub-agent so each agent gets its own ``Trajectory`` -- enabling per-agent
reward assignment for RL training.

Note:
    This uses the "agents as tools" pattern (not handoffs). Handoffs are a
    one-way transfer of control, so the orchestrator would lose the ability
    to call the judge after the solver finishes.  Agents-as-tools keeps the
    orchestrator in charge throughout the conversation.

Prerequisites:
    pip install openai-agents rllm
    export OPENAI_API_KEY=<your-key>

Usage:
    python openai_multi_agent_with_tracing.py
"""

import asyncio
import os
import sys

from agents import Agent, Runner

from rllm.sdk.integrations.openai_agents import RLLMTrajectoryHooks

# ---------------------------------------------------------------------------
# 1. Define sub-agents
# ---------------------------------------------------------------------------

solver = Agent(
    name="solver",
    model="gpt-4o-mini",
    instructions=("You are a math solver. Given a math problem, work through it step by step and provide the final numeric answer on the last line in the format: ANSWER: <number>"),
)

judge = Agent(
    name="judge",
    model="gpt-4o-mini",
    instructions=("You are a math verifier. You will receive a math problem and a proposed solution. Check if the solution is correct. Respond with CORRECT if the answer is right, or INCORRECT with the right answer if it is wrong."),
)

# ---------------------------------------------------------------------------
# 2. Create hooks FIRST, then wire them into both the orchestrator's
#    sub-agent tools and the top-level Runner.run() call.
# ---------------------------------------------------------------------------

hooks = RLLMTrajectoryHooks()

orchestrator = Agent(
    name="orchestrator",
    model="gpt-4o-mini",
    instructions=("You coordinate solving and verifying math problems.\n1. First, call the 'solve_math' tool to get a solution.\n2. Then, call the 'verify_solution' tool with the problem and the proposed solution to verify correctness.\n3. Report the verified answer to the user."),
    tools=[
        solver.as_tool(
            tool_name="solve_math",
            tool_description="Solve a math problem step by step and return the answer.",
            hooks=hooks,
        ),
        judge.as_tool(
            tool_name="verify_solution",
            tool_description="Verify whether a proposed math solution is correct.",
            hooks=hooks,
        ),
    ],
)


# ---------------------------------------------------------------------------
# 3. Run with rLLM tracing and inspect per-agent trajectories
# ---------------------------------------------------------------------------


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set.\n  export OPENAI_API_KEY=sk-...\nGet one at https://platform.openai.com/api-keys")

    print("=== Running multi-agent system with rLLM tracing ===\n")

    result = await Runner.run(
        orchestrator,
        "What is the sum of the first 20 prime numbers?",
        hooks=hooks,
    )

    print(f"Final output:\n{result.final_output}\n")

    # -------------------------------------------------------------------
    # 4. Get the COMBINED trajectory (all agents merged)
    # -------------------------------------------------------------------

    combined = hooks.get_trajectory()
    print(f"{'=' * 60}")
    print("COMBINED TRAJECTORY")
    print(f"{'=' * 60}")
    print(f"  Total LLM calls: {len(combined.steps)}")
    print(f"  Num agents:       {combined.metadata.get('num_llm_calls')}")

    # -------------------------------------------------------------------
    # 5. Get PER-AGENT trajectories for independent reward assignment
    # -------------------------------------------------------------------

    per_agent = hooks.get_trajectories_by_agent()

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
            print(f"  Step {i + 1}: {'[tool_call]' if has_tools else str(content)[:80]}...")

    # -------------------------------------------------------------------
    # 6. Assign rewards per agent
    # -------------------------------------------------------------------

    expected = "639"  # sum of first 20 primes

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

    print(f"\nCollected {len(per_agent)} per-agent trajectories for training.")


if __name__ == "__main__":
    asyncio.run(main())

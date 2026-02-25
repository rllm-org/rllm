"""Example: Google ADK agent with rLLM trajectory tracing.

Shows how to add rLLM trajectory collection to a standard ADK agent with
just a few extra lines.  The collected ``TrajectoryView`` can be used for
SFT distillation or RL training.

Prerequisites:
    pip install google-adk rllm
    export GOOGLE_API_KEY=<your-key>   # or configure Vertex AI credentials

Usage:
    python adk_agent_with_tracing.py
"""

import asyncio
import os
import sys

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from rllm.sdk.integrations.adk import RLLMTrajectoryPlugin

# ---------------------------------------------------------------------------
# 1. Define your ADK agent (standard ADK code -- nothing rLLM-specific)
# ---------------------------------------------------------------------------

agent = Agent(
    name="math_solver",
    model="gemini-2.5-flash",
    instruction=("You are a math tutor. Solve the given problem step by step, then give the final numeric answer on the last line."),
    description="Solves math problems with step-by-step reasoning.",
)

# ---------------------------------------------------------------------------
# 2. Add rLLM tracing -- just create the plugin and pass it to the Runner
# ---------------------------------------------------------------------------

trajectory_plugin = RLLMTrajectoryPlugin()

runner = Runner(
    app_name="math_agent",
    agent=agent,
    session_service=InMemorySessionService(),
    plugins=[trajectory_plugin],
    auto_create_session=True,
)


# ---------------------------------------------------------------------------
# 3. Run the agent and collect the trajectory
# ---------------------------------------------------------------------------


async def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("ERROR: GOOGLE_API_KEY not set.\n  export GOOGLE_API_KEY=<your-key>\nGet one at https://aistudio.google.com/apikey")

    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="What is 15 * 7 + 23?")],
    )

    print("=== Running ADK agent with rLLM tracing ===\n")

    async for event in runner.run_async(
        user_id="user_1",
        session_id="session_1",
        new_message=user_message,
    ):
        if event.content and event.content.parts:
            text = "".join(p.text for p in event.content.parts if p.text)
            if text:
                print(f"[{event.author}] {text}")

    # -----------------------------------------------------------------------
    # 4. Get the trajectory -- ready for reward assignment & training
    # -----------------------------------------------------------------------

    traj = trajectory_plugin.get_trajectory()

    print("\n=== Trajectory Summary ===")
    print(f"Agent:          {traj.name}")
    print(f"LLM calls:      {len(traj.steps)}")
    print(f"User input:     {traj.input}")
    print(f"Final output:   {traj.output[:100] if traj.output else None}...")

    for i, step in enumerate(traj.steps):
        print(f"\n--- Step {i + 1} ---")
        print(f"  Trace ID:  {step.id}")
        n_msgs = len(step.input.get("messages", [])) if isinstance(step.input, dict) else 0
        print(f"  Messages:  {n_msgs}")
        print(f"  Reward:    {step.reward}")

    # -----------------------------------------------------------------------
    # 5. Assign rewards (example: simple correctness check)
    # -----------------------------------------------------------------------

    expected_answer = "128"
    if traj.output and expected_answer in traj.output:
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

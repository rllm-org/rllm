"""Example: Strands agent with rLLM trajectory tracing.

Shows how to add rLLM trajectory collection to a standard Strands agent with
just a few extra lines.  The collected ``Trajectory`` can be used for
SFT distillation or RL training.

Prerequisites:
    pip install 'strands-agents[openai]' rllm
    export OPENAI_API_KEY=<your-key>

Usage:
    python strands_agent_with_tracing.py
"""

import os
import sys

from strands import Agent
from strands.models.openai import OpenAIModel

from rllm.sdk.integrations.strands import RLLMTrajectoryHookProvider

# ---------------------------------------------------------------------------
# 1. Define your Strands agent (standard Strands code -- nothing rLLM-specific)
# ---------------------------------------------------------------------------

model = OpenAIModel(
    client_args={"api_key": os.environ.get("OPENAI_API_KEY", "")},
    model_id="gpt-4o-mini",
)

# ---------------------------------------------------------------------------
# 2. Add rLLM tracing -- just create the hook provider and pass it to Agent
# ---------------------------------------------------------------------------

hook_provider = RLLMTrajectoryHookProvider()

agent = Agent(
    model=model,
    system_prompt="You are a math tutor. Solve the given problem step by step, then give the final numeric answer on the last line.",
    hooks=[hook_provider],
    callback_handler=None,
)


# ---------------------------------------------------------------------------
# 3. Run the agent and collect the trajectory
# ---------------------------------------------------------------------------


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set.\n  export OPENAI_API_KEY=sk-...\nGet one at https://platform.openai.com/api-keys")

    print("=== Running Strands agent with rLLM tracing ===\n")

    result = agent("What is 15 * 7 + 23?")
    print(f"\nAgent output: {result}\n")

    # -------------------------------------------------------------------
    # 4. Get the trajectory -- ready for reward assignment & training
    # -------------------------------------------------------------------

    traj = hook_provider.get_trajectory()

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
    # 5. Assign rewards (example: simple correctness check)
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

    # The Trajectory is now ready for:
    #   - SFT distillation (use traj.steps[i].input/output as training pairs)
    #   - RL training with rLLM (pass to rLLM trainer)
    print(f"\nTrajectory object: {traj.model_dump_json(indent=2)[:500]}...")


if __name__ == "__main__":
    main()

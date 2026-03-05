"""Hello-world ADK agent with Rllm telemetry (stdout backend).

Usage:
    ANTHROPIC_API_KEY=sk-... python examples/hello_world.py
"""

import asyncio
import random

import rllm_telemetry
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

# ----- Tools -----


def roll_die(sides: int) -> int:
    """Roll a die and return the rolled result.

    Args:
        sides: The integer number of sides the die has.

    Returns:
        An integer of the result of rolling the die.
    """
    return random.randint(1, sides)


async def check_prime(nums: list[int]) -> str:
    """Check if a given list of numbers are prime.

    Args:
        nums: The list of numbers to check.

    Returns:
        A str indicating which number is prime.
    """
    primes = set()
    for number in nums:
        number = int(number)
        if number <= 1:
            continue
        is_prime = True
        for i in range(2, int(number**0.5) + 1):
            if number % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.add(number)
    return "No prime numbers found." if not primes else f"{', '.join(str(num) for num in primes)} are prime numbers."


# ----- Agent -----

root_agent = Agent(
    model=LiteLlm(model="anthropic/claude-haiku-4-5-20251001"),
    name="hello_world_agent",
    description="Agent that can roll dice and check prime numbers.",
    instruction=("You roll dice and answer questions about the outcome of the dice rolls. When asked to roll a die, call the roll_die tool. When asked to check primes, call the check_prime tool."),
    tools=[roll_die, check_prime],
)


# ----- Main -----


async def main():
    app_name = "telemetry_test"
    user_id = "user1"

    runner = InMemoryRunner(agent=root_agent, app_name=app_name)

    # Attach Rllm telemetry (stdout backend)
    rllm_telemetry.instrument(runner, backend="stdout")

    session = await runner.session_service.create_session(app_name=app_name, user_id=user_id)

    prompts = [
        "Roll a die with 20 sides",
        "Is that number prime?",
    ]

    for prompt in prompts:
        print(f"\n{'=' * 60}")
        print(f"USER: {prompt}")
        print(f"{'=' * 60}")

        content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"AGENT ({event.author}): {part.text}")


if __name__ == "__main__":
    asyncio.run(main())

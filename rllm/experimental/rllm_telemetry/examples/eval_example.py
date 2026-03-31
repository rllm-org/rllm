"""Example: evaluating an ADK agent with Eval().

Wraps a full ADK agent (with tools, LLM calls, multi-turn if needed)
inside the Eval() task function — the way you'd do it in production.

Usage:
    ANTHROPIC_API_KEY=sk-... python examples/eval_example.py
"""

import rllm_telemetry
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types
from rllm_telemetry import Contains, Eval

# ----- Agent definition -----

root_agent = Agent(
    model=LiteLlm(model="anthropic/claude-haiku-4-5-20251001"),
    name="qa_agent",
    description="A helpful Q&A assistant.",
    instruction=("You are a helpful assistant that answers factual questions concisely. Keep answers to one or two sentences."),
)

# ----- Runner setup -----

APP_NAME = "eval_example"
USER_ID = "eval_user"

runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)

# Attach telemetry so experiment runs also produce full traces
rllm_telemetry.instrument(runner, backend="stdout")


# ----- Task function: wraps the full agent pipeline -----


async def run_agent(input: str) -> str:
    """Send a message through the full ADK agent and return the final response."""
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    content = types.Content(role="user", parts=[types.Part.from_text(text=input)])

    final_text = ""
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text += part.text
    return final_text


# ----- Run the experiment -----

result = Eval(
    "QA Agent - Factual Accuracy",
    data=[
        {
            "input": "What is the capital of France?",
            "expected": "Paris",
        },
        {
            "input": "What language is primarily spoken in Brazil?",
            "expected": "Portuguese",
        },
        {
            "input": "What is the largest planet in our solar system?",
            "expected": "Jupiter",
        },
        {
            "input": "Who wrote Romeo and Juliet?",
            "expected": "Shakespeare",
        },
    ],
    task=run_agent,
    scores=[Contains()],
    metadata={"model": "claude-haiku-4-5-20251001", "description": "Baseline factual QA"},
    backend="stdout",  # emit experiment records via stdout exporter
)

# Programmatic access to results
for case in result.cases:
    print(f"  Input: {case.input}")
    print(f"  Output: {case.output[:80]}..." if len(str(case.output)) > 80 else f"  Output: {case.output}")
    print(f"  Scores: {', '.join(f'{s.name}={s.value:.1f}' for s in case.scores)}")
    print()

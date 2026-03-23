"""End-to-end experiment: comparing two system prompts on a multi-step research agent.

Two agents with identical tools but different system prompts are evaluated
on the same set of questions. Each question requires the agent to use
multiple tools (knowledge base lookup, calculator) to arrive at an answer.
Results are scored with LlmJudge and compared side-by-side.

Usage:
    # Requires an LLM API key and rllm-ui running locally
    ANTHROPIC_API_KEY=sk-... \
    RLLM_API_KEY=your-key \
    python examples/experiment_comparison.py

    # Override agent endpoint (optional, defaults to staging):
    AGENT_ENDPOINT=https://rllm-ui-rllm-staging.up.railway.app \
    python examples/experiment_comparison.py
"""

import asyncio
import math
import os
from types import SimpleNamespace

import rllm_telemetry
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types
from rllm_telemetry import AsyncEval, LlmJudge, compare, export_trajectories

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "anthropic/claude-haiku-4-5-20251001"
JUDGE_MODEL = "anthropic/claude-haiku-4-5-20251001"

APP_NAME = "experiment_comparison"
USER_ID = "eval_user"

AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "")  # empty = use RllmConfig default

# ---------------------------------------------------------------------------
# Tools — shared by both agents
# ---------------------------------------------------------------------------

# Small in-memory knowledge base (simulates a retrieval tool)
_KNOWLEDGE_BASE = {
    "france": "France has a population of 67.75 million (2024). Capital: Paris. GDP: $3.05 trillion. Area: 643,801 km². Currency: Euro. Major industries: tourism, aerospace, luxury goods, agriculture.",
    "japan": "Japan has a population of 123.3 million (2024). Capital: Tokyo. GDP: $4.23 trillion. Area: 377,975 km². Currency: Yen. Major industries: automotive, electronics, robotics, steel.",
    "brazil": "Brazil has a population of 216.4 million (2024). Capital: Brasília. GDP: $2.13 trillion. Area: 8,515,767 km². Currency: Real. Major industries: agriculture, mining, oil, manufacturing.",
    "germany": "Germany has a population of 84.48 million (2024). Capital: Berlin. GDP: $4.46 trillion. Area: 357,022 km². Currency: Euro. Major industries: automotive, machinery, chemicals, electronics.",
    "india": "India has a population of 1.44 billion (2024). Capital: New Delhi. GDP: $3.94 trillion. Area: 3,287,263 km². Currency: Rupee. Major industries: IT services, textiles, agriculture, pharmaceuticals.",
    "renewable energy": "Global renewable energy capacity reached 3,870 GW in 2023. Solar: 1,419 GW. Wind: 1,017 GW. Hydropower: 1,267 GW. Growth rate: 14% year-over-year. China leads with 1,300 GW installed.",
    "electric vehicles": "Global EV sales reached 14.2 million units in 2023, up 35% from 2022. Market share: 18% of all new cars. China: 8.1 million, Europe: 3.2 million, US: 1.4 million. Average EV price: $55,000.",
    "space exploration": "As of 2024: 7,702 active satellites orbit Earth. SpaceX has launched 330+ missions. NASA's Artemis program targets lunar return by 2026. Mars sample return planned for 2033. ISS has been continuously occupied since 2000.",
}


def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information on a topic.

    Args:
        query: The search query — a topic or keyword to look up.

    Returns:
        Relevant information from the knowledge base, or a not-found message.
    """
    query_lower = query.lower()
    results = []
    for key, value in _KNOWLEDGE_BASE.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            results.append(value)
    if not results:
        # Fuzzy: check if any KB entry mentions terms from the query
        for key, value in _KNOWLEDGE_BASE.items():
            if any(word in value.lower() for word in query_lower.split() if len(word) > 3):
                results.append(value)
    return "\n\n".join(results) if results else f"No information found for: {query}"


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression to evaluate (e.g. "67.75 / 643801 * 1000000").

    Returns:
        The computed result as a string.
    """
    # Safe subset of math operations
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "pow": pow,
        "log": math.log,
        "log10": math.log10,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# System prompts — the independent variable
# ---------------------------------------------------------------------------

PROMPT_MINIMAL = "You are a helpful assistant. Use the provided tools to answer the user's question. Give a short, direct answer."

PROMPT_DETAILED = "You are a meticulous research analyst. Follow these steps for every question:\n1. SEARCH: Use search_knowledge_base to find all relevant facts.\n2. COMPUTE: If the question involves numbers or comparisons, use calculate to do the math.\n3. SYNTHESIZE: Combine your findings into a clear, precise answer.\nAlways cite specific numbers and facts from your research. Show your reasoning when making comparisons or calculations."

# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

TOOLS = [search_knowledge_base, calculate]


def make_agent(name: str, instruction: str) -> Agent:
    return Agent(
        model=LiteLlm(model=MODEL),
        name=name,
        description="Research assistant with knowledge base and calculator tools.",
        instruction=instruction,
        tools=TOOLS,
    )


# ---------------------------------------------------------------------------
# Task function — wraps the full ADK agent loop
# ---------------------------------------------------------------------------


def make_task(runner: InMemoryRunner):
    """Create a task function bound to a specific runner."""

    async def run_agent(input: str) -> str:
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        content = types.Content(role="user", parts=[types.Part.from_text(text=input)])

        final_text = ""
        async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=content):
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_text += part.text
        return final_text

    return run_agent


# ---------------------------------------------------------------------------
# Test cases — questions requiring multi-step tool use
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "input": "What is the population density of France (people per km²)?",
        "expected": "About 105 people per km². France has 67.75 million people and an area of 643,801 km².",
    },
    {
        "input": "Which country has a higher GDP per capita, Japan or Germany? By how much?",
        "expected": "Germany has a higher GDP per capita (~$52,800) vs Japan (~$34,300). Germany's is about $18,500 higher.",
    },
    {
        "input": "How many more EV units were sold in China than in Europe and the US combined in 2023?",
        "expected": "China sold 8.1 million EVs. Europe (3.2M) + US (1.4M) = 4.6M. China sold 3.5 million more.",
    },
    {
        "input": "What percentage of global renewable energy capacity is solar power?",
        "expected": "Solar is about 36.7% of global renewable capacity (1,419 GW out of 3,870 GW total).",
    },
    {
        "input": "If India's GDP grows by 7% next year, what would it be? How would it compare to Germany's current GDP?",
        "expected": "India's GDP at 7% growth: ~$4.22 trillion. Still below Germany's $4.46 trillion by about $240 billion.",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    # Build agents with different prompts
    agent_minimal = make_agent("minimal_agent", PROMPT_MINIMAL)
    agent_detailed = make_agent("detailed_agent", PROMPT_DETAILED)

    runner_minimal = InMemoryRunner(agent=agent_minimal, app_name=APP_NAME)
    runner_detailed = InMemoryRunner(agent=agent_detailed, app_name=APP_NAME)

    # Attach observability telemetry (stdout + stream spans to rllm_ui → ClickHouse)
    plugin_minimal = rllm_telemetry.instrument(
        runner_minimal,
        backend="stdout",
        agent_endpoint=AGENT_ENDPOINT,
    )
    plugin_detailed = rllm_telemetry.instrument(
        runner_detailed,
        backend="stdout",
        agent_endpoint=AGENT_ENDPOINT,
    )
    print(f"[debug] Telemetry endpoint: {AGENT_ENDPOINT}")
    print(f"[debug] Exporter type: {type(plugin_minimal._exporter).__name__}")

    # LLM judge scorer
    judge = LlmJudge(
        model=JUDGE_MODEL,
        prompt_template="factuality",
        name="Factuality",
    )

    # Run experiments
    print("Running baseline (minimal prompt)...")
    baseline = await AsyncEval(
        "Minimal Prompt",
        data=TEST_CASES,
        task=make_task(runner_minimal),
        scores=[judge],
        metadata={"prompt_style": "minimal", "model": MODEL},
    )

    print("\nRunning candidate (detailed prompt)...")
    candidate = await AsyncEval(
        "Detailed Prompt",
        data=TEST_CASES,
        task=make_task(runner_detailed),
        scores=[judge],
        metadata={"prompt_style": "detailed", "model": MODEL},
    )

    # Compare
    diff = compare(baseline, candidate)
    diff.print_report()

    # Print detailed case-by-case outputs
    print(f"\n{'=' * 60}")
    print("Detailed Case Outputs")
    print(f"{'=' * 60}")
    for b_case, c_case, case_diff in zip(baseline.cases, candidate.cases, diff.case_diffs, strict=False):
        print(f"\nQ: {b_case.input}")
        print(f"Expected: {b_case.expected}")
        print(f"\n  [Minimal]  (score={b_case.scores[0].value:.2f})")
        b_out = str(b_case.output)
        print(f"  {b_out[:200]}{'...' if len(b_out) > 200 else ''}")
        print(f"\n  [Detailed] (score={c_case.scores[0].value:.2f})")
        c_out = str(c_case.output)
        print(f"  {c_out[:200]}{'...' if len(c_out) > 200 else ''}")

        delta = case_diff.score_diffs.get("Factuality", 0)
        if delta > 0:
            label = "IMPROVED"
        elif delta < 0:
            label = "REGRESSED"
        else:
            label = "UNCHANGED"
        print(f"\n  Delta: {delta:+.2f} ({label})")
        print(f"  {'-' * 50}")

    # -----------------------------------------------------------------
    # Export trajectories — build trajectory-like objects from results
    # -----------------------------------------------------------------
    trajectories = []
    for result, prompt_label in [(baseline, "minimal"), (candidate, "detailed")]:
        for case in result.cases:
            step = SimpleNamespace(
                input=case.input,
                output=case.output,
                reward=case.scores[0].value if case.scores else 0.0,
                done=True,
                metadata={"expected": case.expected},
            )
            traj = SimpleNamespace(
                uid=case.case_id,
                name=f"{prompt_label}_agent",
                task={"input": case.input, "expected": case.expected},
                steps=[step],
                reward=case.scores[0].value if case.scores else 0.0,
                signals={s.name: s.value for s in case.scores},
                metadata=result.record.metadata,
            )
            trajectories.append(traj)

    export_trajectories(trajectories)

    # Close telemetry plugins so agent sessions are marked as completed
    await plugin_minimal.close()
    await plugin_detailed.close()


if __name__ == "__main__":
    asyncio.run(main())

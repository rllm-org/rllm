r"""Single-turn math agent.

The agent receives a math problem and returns reasoning followed by a
final answer in ``\boxed{...}`` notation, in a single LLM call. The
full assistant response is stored on the episode so the evaluator
extracts the boxed answer and grades it against ground truth.

This is the no-tool counterpart to ``cookbooks/math_tool_agent/`` and
covers the workload of the legacy ``examples/{deepscaler, simple_math,
gsm8k_lora, math_tinker}`` examples in one cookbook.

Task metadata schema (from rllm dataset pull math500 / hendrycks_math
/ gsm8k / deepscaler_math)::

    {
        "question" or "problem": str,   # the problem statement
        "ground_truth" or "answer": str, # boxed answer or numeric value
        ...
    }
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a math expert. Solve the problem step by step, showing your reasoning
clearly. Put your final answer inside \\boxed{} notation.

Example: "After simplifying, the answer is \\boxed{42}."

Only the contents of the LAST \\boxed{...} in your reply will be graded.
"""


@rllm.rollout(name="math")
async def math_flow(task: Task, config: AgentConfig) -> Episode:
    """One-shot math flow: LLM emits reasoning + boxed answer; evaluator grades."""
    meta = task.metadata or {}
    question = str(meta.get("question") or meta.get("problem") or task.instruction or "")

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    try:
        resp = await client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.sampling_params.get("temperature", 0.6),
            max_tokens=config.sampling_params.get("max_tokens", 8192),
            timeout=300,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("math task %s: LLM call failed: %s", task.id, e)
        content = ""

    messages.append({"role": "assistant", "content": content})
    step = Step(
        chat_completions=list(messages),
        model_response=content,
        action=content,
        thought=content,
    )

    # Store the raw model response — the evaluator extracts \boxed{...}
    # via rllm.rewards.math_utils.utils.extract_answer.
    return Episode(
        trajectories=[Trajectory(name="math", steps=[step])],
        artifacts={"answer": content},
    )

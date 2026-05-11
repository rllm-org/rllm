"""Solver-Judge AgentFlow — multi-agent countdown solver.

A solver generates N candidate solutions in parallel, then a judge
selects the best one. Uses plain OpenAI client — works identically
for eval and training (the gateway handles trace capture).
"""

from __future__ import annotations

import asyncio
import re

from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

N_SOLUTIONS = 2


@rllm.rollout(name="solver-judge")
async def solver_judge_flow(task: Task, config: AgentConfig) -> Episode:
    """AgentFlow: solver generates N solutions, judge picks the best."""
    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    problem = _format_countdown_problem(task)

    # Step 1: Solver generates N solutions in parallel
    solver_trajectories = await _generate_solutions(client, config.model, problem)

    # Step 2: Judge selects the best solution
    solutions = [t.steps[0].action for t in solver_trajectories]
    judge_trajectory = await _judge_solutions(client, config.model, problem, solutions)

    selected = judge_trajectory.steps[0].action
    return Episode(
        trajectories=[*solver_trajectories, judge_trajectory],
        artifacts={"answer": selected},
    )


async def _generate_solutions(client: AsyncOpenAI, model: str, problem: str) -> list[Trajectory]:
    async def _solve() -> Trajectory:
        messages = [{"role": "user", "content": f"{problem}. Output the final answer within <answer>...</answer>"}]
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            max_tokens=2048,
        )
        content = response.choices[0].message.content or ""
        parsed = _parse_answer(content)
        return Trajectory(
            name="solver",
            steps=[
                Step(
                    chat_completions=messages + [{"role": "assistant", "content": content}],
                    model_response=content,
                    action=parsed,
                )
            ],
        )

    return await asyncio.gather(*(_solve() for _ in range(N_SOLUTIONS)))


async def _judge_solutions(client: AsyncOpenAI, model: str, problem: str, solutions: list[str]) -> Trajectory:
    prompt = _create_judge_prompt(problem, solutions)
    messages = [{"role": "user", "content": prompt}]
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        max_tokens=2048,
    )
    content = response.choices[0].message.content or ""
    selected = _parse_judge_response(content, solutions)
    return Trajectory(
        name="judge",
        steps=[
            Step(
                chat_completions=messages + [{"role": "assistant", "content": content}],
                model_response=content,
                action=selected,
            )
        ],
    )


# -- Parsing helpers --------------------------------------------------------


def _format_countdown_problem(task: Task) -> str:
    """Render the countdown task into a natural-language problem statement.

    The countdown DatasetRegistry transform returns only ``{target, nums,
    data_source}`` (no ``question`` field), so ``task.instruction`` is empty
    and the flow has to format the prompt itself from metadata.
    """
    md = task.metadata or {}
    target = md.get("target")
    nums = md.get("nums")
    if target is None or nums is None:
        return str(task.instruction or "")
    return f"Using the numbers {list(nums)}, write an arithmetic expression that evaluates to {target}. Each number must be used exactly once and only +, -, *, / are allowed."


def _parse_answer(response: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
    if match:
        return f"<answer>{match.group(1).strip()}</answer>"
    return "No solution found"


def _parse_judge_response(response: str, solutions: list[str]) -> str:
    match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
    if match:
        try:
            idx = int(match.group(1).strip())
            return solutions[idx - 1]
        except (ValueError, IndexError):
            return ""
    return ""


def _create_judge_prompt(problem: str, solutions: list[str]) -> str:
    prompt = f"""You are an expert verifier. Given a countdown problem and multiple solution attempts, select a correct solution.
Problem:
{problem}
Solutions to evaluate:
"""
    for i, solution in enumerate(solutions, 1):
        prompt += f"\nSolution {i}:\n{solution}\n"

    prompt += """
A correct solution must satisfy the following criteria:
1. The solution uses only the given numbers.
2. Each number is used exactly once.
3. Only basic arithmetic operations (+, -, *, /) are used.
4. The calculation results in the target number.
5. The final answer is clearly marked within <answer>...</answer> tags.
Output the index of your selected solution within <answer>...</answer> tags, e.g., <answer>1</answer> for the first solution, <answer>2</answer> for the second solution, etc. If multiple solutions are correct, output the index of the first correct solution."""
    return prompt

"""Solver-Judge AgentFlow — multi-agent countdown solver.

A solver generates N candidate solutions in parallel, then a judge
selects the best one. Uses plain OpenAI client — works identically
for eval and training (the gateway handles trace capture).
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

import rllm
from rllm.experimental.eval.types import AgentConfig, Task
from rllm.types import Episode, Step, Trajectory

N_SOLUTIONS = 2


@rllm.rollout(name="solver-judge")
def solver_judge_flow(task: Task, config: AgentConfig) -> Episode:
    """AgentFlow: solver generates N solutions, judge picks the best."""
    data = task.data
    client = OpenAI(base_url=config.base_url, api_key="EMPTY")
    problem = data.get("question", "")

    # Step 1: Solver generates N solutions in parallel
    solver_trajectories = _generate_solutions(client, config.model, problem)

    # Step 2: Judge selects the best solution
    solutions = [t.steps[0].action for t in solver_trajectories]
    judge_trajectory = _judge_solutions(client, config.model, problem, solutions)

    selected = judge_trajectory.steps[0].action
    return Episode(
        trajectories=[*solver_trajectories, judge_trajectory],
        artifacts={"answer": selected},
    )


def _generate_solutions(client: OpenAI, model: str, problem: str) -> list[Trajectory]:
    """Generate N solutions in parallel using threads."""

    def _solve() -> Trajectory:
        messages = [{"role": "user", "content": f"{problem}. Output the final answer within <answer>...</answer>"}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            max_tokens=1000,
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

    with ThreadPoolExecutor(max_workers=N_SOLUTIONS) as pool:
        futures = [pool.submit(_solve) for _ in range(N_SOLUTIONS)]
        return [f.result() for f in futures]


def _judge_solutions(client: OpenAI, model: str, problem: str, solutions: list[str]) -> Trajectory:
    prompt = _create_judge_prompt(problem, solutions)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        max_tokens=1000,
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

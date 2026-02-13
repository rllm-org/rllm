"""Solver-Judge agent logic adapted for the remote training framework.

This implements the same Solver-Judge workflow as
``examples/solver_judge/solver_judge_flow.py`` but uses
``RemoteRolloutClient.chat_completion()`` instead of ``RolloutEngine``.

The workflow:
1. Solver generates ``n_solutions`` candidate solutions in parallel.
2. Each solution is scored with a local reward function.
3. Judge reviews all candidates and selects the best one.
4. The final trajectory carries the judge's reward.
"""

from __future__ import annotations

import asyncio
import random
import re

from rllm.experimental.fully_async.protocol import Trajectory


# ── Reward function (runs locally, no GPU needed) ───────────────────────────


def extract_answer(text: str) -> str | None:
    """Extract content from <answer>...</answer> tags."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def countdown_reward(task: dict, solution_text: str) -> float:
    """Score a Countdown solution. Returns 1.0 if correct, 0.0 otherwise."""
    target = task.get("target")
    nums = task.get("nums", [])
    if target is None or not nums:
        return 0.0

    equation = extract_answer(solution_text)
    if equation is None:
        return 0.0

    # Validate: equation must use exactly the available numbers
    try:
        numbers_in_eq = sorted(int(n) for n in re.findall(r"\d+", equation))
        if numbers_in_eq != sorted(nums):
            return 0.0
    except Exception:
        return 0.0

    # Evaluate safely
    try:
        if not re.match(r"^[\d+\-*/().\s]+$", equation):
            return 0.0
        result = eval(equation, {"__builtins__": None}, {})  # noqa: S307
        return 1.0 if abs(result - target) < 1e-5 else 0.0
    except Exception:
        return 0.0


# ── Solver ──────────────────────────────────────────────────────────────────


async def _solve(client, problem: str, sampling_params: dict | None = None):
    """Generate one solver response. Returns (message_text, OutputWithVersion)."""
    messages = [
        {
            "role": "user",
            "content": (
                f"{problem}. Output the final answer within "
                f"<answer>...</answer>"
            ),
        }
    ]
    params = sampling_params or {"temperature": 1.0, "max_new_tokens": 1024}
    msg, output = await client.chat_completion(messages, sampling_params=params)
    return msg.get("content", ""), output


async def solve_n(client, problem: str, n: int = 2, sampling_params: dict | None = None):
    """Generate *n* solver solutions in parallel."""
    tasks = [_solve(client, problem, sampling_params) for _ in range(n)]
    return await asyncio.gather(*tasks)


# ── Judge ───────────────────────────────────────────────────────────────────


def _build_judge_prompt(problem: str, solutions: list[str]) -> str:
    prompt = (
        "You are an expert verifier. Given a countdown problem and multiple "
        "solution attempts, select a correct solution.\n\n"
        f"Problem:\n{problem}\n\n"
        "Solutions to evaluate:\n"
    )
    for i, sol in enumerate(solutions, 1):
        prompt += f"\nSolution {i}:\n{sol}\n"
    prompt += (
        "\nA correct solution must satisfy the following criteria:\n"
        "1. The solution uses only the given numbers.\n"
        "2. Each number is used exactly once.\n"
        "3. Only basic arithmetic operations (+, -, *, /) are used.\n"
        "4. The calculation results in the target number.\n"
        "5. The final answer is clearly marked within <answer>...</answer> tags.\n"
        "\nOutput the index of your selected solution within <answer>...</answer> "
        "tags, e.g., <answer>1</answer> for the first solution."
    )
    return prompt


def _parse_judge_selection(judge_text: str, solutions: list[str]) -> str:
    """Parse the judge's selection index and return the chosen solution text."""
    answer = extract_answer(judge_text)
    if answer is None:
        return ""
    try:
        idx = int(answer) - 1  # 1-indexed -> 0-indexed
        return solutions[idx] if 0 <= idx < len(solutions) else ""
    except (ValueError, IndexError):
        return ""


async def judge(client, problem: str, solutions: list[str], sampling_params: dict | None = None):
    """Run the judge on a list of candidate solutions."""
    messages = [{"role": "user", "content": _build_judge_prompt(problem, solutions)}]
    params = sampling_params or {"temperature": 1.0, "max_new_tokens": 1024}
    msg, output = await client.chat_completion(messages, sampling_params=params)
    judge_text = msg.get("content", "")
    selected = _parse_judge_selection(judge_text, solutions)
    return judge_text, selected, output


# ── Rollout function ────────────────────────────────────────────────────────


async def rollout_fn(client, tokenizer, n_solutions: int = 2, **kwargs):
    """Full Solver-Judge rollout for one Countdown task.

    This is the ``rollout_fn`` passed to :class:`AgentTrainerClient`.

    Args:
        client: ``RemoteRolloutClient`` instance.
        tokenizer: HuggingFace tokenizer (loaded locally).
        n_solutions: Number of solver candidates to generate.
        **kwargs: A single row from the Countdown dataset
                  (keys: question, target, nums, ground_truth).

    Returns:
        :class:`Trajectory` with sequences from all model calls and the
        judge's reward.
    """
    task = kwargs  # contains question, target, nums, ground_truth
    problem = task["question"]
    param_version_start = client.cur_version

    # ── Step 1: Solver generates n candidates ──
    solver_results = await solve_n(client, problem, n=n_solutions)
    solver_texts = []
    solver_sequences = []
    solver_rewards = []
    for text, output in solver_results:
        solver_texts.append(text)
        solver_sequences.append(output.to_sequence())
        solver_rewards.append(countdown_reward(task, text))

    # ── Step 2: Judge selects the best ──
    judge_text, selected_solution, judge_output = await judge(
        client, problem, solver_texts,
    )
    judge_sequence = judge_output.to_sequence()

    # Reward: did the judge pick a correct solution?
    judge_reward = countdown_reward(task, selected_solution)

    # ── Build trajectory ──
    # Include all sequences (solver + judge) so the trainer sees the full
    # token-level data.  The trajectory reward is the judge's reward.
    all_sequences = solver_sequences + [judge_sequence]
    trajectory = Trajectory(
        sequences=all_sequences,
        reward=judge_reward,
        metadata={
            "solver_rewards": solver_rewards,
            "solver_avg_acc": sum(solver_rewards) / len(solver_rewards),
            "judge_acc": judge_reward,
            "judge_selected": selected_solution[:200],
            "param_version_start": param_version_start,
            "param_version_end": client.cur_version,
        },
    )

    if random.random() < 0.005:
        print(f"\n{'='*60}")
        print(f"Problem: {problem[:120]}")
        print(f"Solver rewards: {solver_rewards}")
        print(f"Judge selected: {selected_solution[:120]}")
        print(f"Judge reward: {judge_reward}")
        print(f"{'='*60}\n")

    return trajectory

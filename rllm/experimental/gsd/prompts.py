"""GSD prompt templates for math training.

Defines the four prompt types used by :class:`GsdWorkflow`:

* **Hint prompt** — asks the model (as a strategist) to generate a
  solution-independent hint for a math problem.
* **Teacher prompt** — the student prompt augmented with the hint.
* **Student prompt** — the plain problem without any hint.
* **Hint extraction** — parses the hint from the model's response.
"""

from __future__ import annotations

import re

HINT_SYSTEM_PROMPT = """
You are a math problem advisor. Given a problem, provide a brief, high-level \
hint to guide a solver. Do NOT solve the problem or state the answer.

Rules:
- Give at most 3 short bullet points.
- Each bullet should be one sentence.
- Stay general — suggest directions, not specific steps.

Example:

Problem: Find the number of integers n with 1 <= n <= 2023 such that n^2 + n is divisible by 6.

<hint>
- Consider what n^2 + n factors as and what divisibility by 6 requires.
- Check small cases to identify a repeating pattern modulo 6.
- Count how many complete cycles fit in the range.
</hint>
""".strip()

SOLVER_SYSTEM_PROMPT = """
You are an expert mathematician. Solve the given problem step by step, \
showing your reasoning clearly. Put your final answer in \\boxed{}.
""".strip()

SOLVE_INSTRUCTION = "Please solve step by step. Put your final answer in \\boxed{}."


def build_hint_prompt(
    question: str,
    experiences: list[dict] | None = None,
) -> list[dict]:
    """Build the self-hinting prompt.

    Args:
        question: The math problem statement.
        experiences: Optional list of past experience dicts with keys
            ``"text"``, ``"hint"``, ``"summary"``.  For Phase 1 this is
            ``None`` (no buffer retrieval).
    """
    user_parts: list[str] = []

    if experiences:
        user_parts.append("[Past Experience]")
        for i, exp in enumerate(experiences, 1):
            user_parts.append(f"--- Experience {i} ---")
            user_parts.append(f"Problem: {exp.get('text', '')}")
            user_parts.append(f"Hint used: {exp.get('hint', '')}")
            user_parts.append(f"Outcome: {exp.get('summary', '')}")
        user_parts.append("")

    user_parts.append(f"Problem: {question}")
    user_parts.append("")
    user_parts.append("Provide a brief hint in <hint>...</hint> tags (at most 3 bullet points).")

    return [
        {"role": "system", "content": HINT_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def build_teacher_prompt(question: str, hint: str) -> list[dict]:
    """Build the teacher prompt — solver system prompt + hint + problem."""
    return [
        {
            "role": "system",
            "content": (f"{SOLVER_SYSTEM_PROMPT}\n\nUse the following strategic hint to guide your approach:\n\n{hint}"),
        },
        {"role": "user", "content": question},
    ]


def build_student_prompt(question: str) -> list[dict]:
    """Build the student prompt — solver system prompt + plain problem."""
    return [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


_HINT_RE = re.compile(r"<hint>(.*?)</hint>", re.DOTALL | re.IGNORECASE)


def extract_hint(response: str) -> str:
    """Extract hint text from ``<hint>...</hint>`` tags, or return raw response."""
    match = _HINT_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()

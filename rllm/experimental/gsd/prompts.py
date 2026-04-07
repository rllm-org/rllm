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
You are a competition math strategist. Given a new problem and optionally \
experiences from similar past problems, generate a strategic hint that will \
help approach this problem.

IMPORTANT:
- Do NOT attempt to solve the problem.
- Do NOT guess or state the answer.
- Focus on: identifying the problem type, relevant techniques, common \
pitfalls, and a high-level approach strategy.
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
            user_parts.append(f"Strategy used: {exp.get('hint', '')}")
            user_parts.append(f"Outcome: {exp.get('summary', '')}")
    else:
        user_parts.append("[No Past Experience Available Now. Use Your Own Knowledge and Intuition to Generate a Hint.]")
    user_parts.append("")

    user_parts.append(f"[Current Problem]\n{question}")
    user_parts.append("")
    user_parts.append(
        "Generate a strategic hint including: problem type, relevant techniques, pitfalls to watch out for, and a suggested high-level approach. Enclose your hint in <hint>...</hint> tags."
    )

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

r"""Multi-turn iterative-coding agent (deepcoder).

The agent receives a competition-style coding problem, writes code in a
``\`\`\`python`` block, and runs the model's solution against hidden test
cases. If any test fails, the test results are fed back as a user
message and the model gets another turn to revise. Loop ends on first
all-pass turn or after ``max_turns`` revisions.

The test runner is :class:`rllm.rewards.code_reward.RewardCodeFn` —
the same code path that powers ``rllm.eval.reward_fns.code``. Reusing
it means the in-loop feedback grader is identical to the final
evaluator, so the agent never sees a different signal at train-time
vs eval-time.

Task metadata schema (produced by ``prepare_data.py``)::

    {
        "question": str,       # full problem statement (system + problem text)
        "ground_truth": str,   # JSON-encoded list of test cases
        "data_source": str,    # "livecodebench" / "codeforces" / "taco" / "apps" / ...
        "starter_code": str,   # optional template
        "metadata": str,       # JSON-encoded extra metadata (func_name, ...)
    }
"""

from __future__ import annotations

import logging
import re

from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a competitive programmer. You will be given a coding problem and must write a Python program that solves it.

Your code MUST be wrapped in a fenced code block with the language tag, e.g.

```python
# your solution here
```

Only the contents of the LAST fenced ```python``` block in your reply will be
executed against the hidden tests. Read the problem carefully — the test
format (stdin/stdout vs function call) is implied by the problem statement
and any starter code.
"""


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    r"""Pull the contents of the last ``\`\`\`python`` block in *text*."""
    matches = _CODE_BLOCK_RE.findall(text)
    return matches[-1].strip() if matches else None


def _truncate(s: str, n: int = 240) -> str:
    s = str(s)
    if len(s) <= n:
        return s
    return s[: n // 2] + "  …(truncated)…  " + s[-n // 2 :]


def format_feedback(reward_metadata: dict, max_failures: int = 2) -> str:
    """Render RewardCodeFn metadata into a user-facing feedback message.

    Mirrors the legacy ``CompetitionCodingAgent.format_test_results`` style:
    show up to ``max_failures`` failing tests with input/expected/actual,
    and ask for a revision.
    """
    test_results = reward_metadata.get("test_results", []) or []
    failures = [t for t in test_results if isinstance(t, dict) and not t.get("passed", False)]

    if not failures:
        return "All visible tests passed, but a hidden check failed. Re-examine your solution for edge cases (empty inputs, large values, off-by-one errors) and submit a revised version."

    lines = ["Some test cases failed. Here are the failures:\n"]
    for i, t in enumerate(failures[:max_failures], start=1):
        lines.append(f"### Test {i}")
        if "input" in t:
            lines.append(f"Input: {_truncate(t['input'])}")
        if "expected" in t:
            lines.append(f"Expected: {_truncate(t['expected'])}")
        if t.get("output") is not None:
            lines.append(f"Got: {_truncate(t['output'])}")
        if t.get("error_message"):
            lines.append(f"Error: {_truncate(t['error_message'])}")
        lines.append("")
    lines.append("Analyze the failures, then output a revised complete solution in a single ```python``` block.")
    return "\n".join(lines)


@rllm.rollout(name="deepcoder")
async def deepcoder_flow(task: Task, config: AgentConfig) -> Episode:
    """Iterative-coding flow: write → test → feedback → revise."""
    from rllm.rewards.code_reward import RewardCodeFn
    from rllm.rewards.reward_types import RewardConfig

    meta = task.metadata or {}
    question = str(meta.get("question") or task.instruction or "")
    max_turns = int(meta.get("max_turns", 3))

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    grader = RewardCodeFn(RewardConfig())

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    steps: list[Step] = []
    last_code: str | None = None
    passed = False

    for turn in range(max_turns):
        try:
            resp = await client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=0.6,
                max_tokens=8192,
                timeout=300,
            )
        except Exception as e:
            logger.warning("deepcoder task %s turn %d: LLM call failed: %s", task.id, turn, e)
            break

        content = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})

        code = extract_code(content)
        last_code = code if code is not None else last_code

        steps.append(
            Step(
                chat_completions=list(messages),
                model_response=content,
                action=code,
                thought=content,
            )
        )

        if code is None:
            messages.append(
                {
                    "role": "user",
                    "content": ("Your reply did not contain a fenced ```python``` code block. Output your full solution wrapped in ```python ... ```."),
                }
            )
            continue

        # Grade against hidden tests using the same reward fn the evaluator uses.
        reward = grader(task_info=meta, action=code)
        steps[-1].reward = float(reward.reward)
        if reward.is_correct:
            passed = True
            break

        # Last turn: don't bother with feedback — the loop is about to end.
        if turn == max_turns - 1:
            break

        messages.append({"role": "user", "content": format_feedback(reward.metadata)})

    return Episode(
        trajectories=[Trajectory(name="deepcoder", steps=steps)],
        artifacts={"answer": last_code or "", "passed": passed, "turns": len(steps)},
        is_correct=passed,
    )

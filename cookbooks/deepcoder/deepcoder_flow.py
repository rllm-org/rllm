r"""Single-turn deepcoder agent.

The agent receives a competition-style coding problem and returns Python
code in a fenced ``\`\`\`python`` block in a single LLM call. Long-chain
reasoning happens *inside* the assistant message (e.g. ``<think>…</think>``
or just step-by-step prose before the final code block) — there is no
multi-turn revise/feedback loop. The full assistant response is stored
in ``episode.artifacts["answer"]`` so the evaluator can pull the last
fenced code block out exactly the way :class:`RewardCodeFn` expects.

Task metadata schema (produced by ``prepare_deepcoder_data.py``)::

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

from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a competitive programmer. You will be given a coding problem and must
write a Python program that solves it.

Reason step by step, then put your final, complete solution in a single fenced
code block:

```python
# your solution here
```

Only the contents of the LAST fenced ```python``` block in your reply will be
executed against the hidden tests. Read the problem carefully — the test
format (stdin/stdout vs function call) is implied by the problem statement
and any starter code.
"""


@rllm.rollout(name="deepcoder")
async def deepcoder_flow(task: Task, config: AgentConfig) -> Episode:
    """One-shot coding flow: LLM emits a single response, evaluator grades."""
    meta = task.metadata or {}
    question = str(meta.get("question") or task.instruction or "")

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    try:
        resp = await client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=0.6,
            max_tokens=16384,
            timeout=600,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("deepcoder task %s: LLM call failed: %s", task.id, e)
        content = ""

    messages.append({"role": "assistant", "content": content})
    step = Step(
        chat_completions=list(messages),
        model_response=content,
        action=content,
        thought=content,
    )

    # Store the raw model response — the evaluator's RewardCodeFn extracts
    # the last fenced ``\`\`\`python`` block via extract_code_from_model.
    return Episode(
        trajectories=[Trajectory(name="deepcoder", steps=[step])],
        artifacts={"answer": content},
    )

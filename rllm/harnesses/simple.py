"""SimpleHarness: one-shot LLM call for data tasks.

Used by catalog datasets (gsm8k, MATH, MMLU, etc.) where the agent's
"work" is a single chat completion. Sets ``trajectory.output`` to the
LLM response so downstream score_fns can extract the answer.

Implements the rLLM ``AgentFlow`` protocol with no sandbox dependency.
"""

from __future__ import annotations

import logging

from rllm.task import Task
from rllm.tasks.harness import register_harness
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)


_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer the question to the best of your ability."


class SimpleHarness:
    """One-shot LLM harness for data tasks (gsm8k-style).

    No sandbox required. Pure ``AgentFlow`` — produces an Episode with
    a single Trajectory whose ``output`` is the LLM response.
    """

    name = "simple"
    max_concurrent = 64

    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    def run(self, task: Task, config) -> Episode:
        from openai import OpenAI

        client = OpenAI(base_url=config.base_url, api_key="EMPTY")

        instruction = task.instruction
        # Multimodal: pass through the content blocks as-is
        if isinstance(instruction, list):
            user_content = instruction
        else:
            user_content = str(instruction)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response = client.chat.completions.create(model=config.model, messages=messages)
            answer = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("SimpleHarness LLM call failed for task %s: %s", task.id, e)
            answer = ""

        step = Step(
            id="step-0",
            input=str(instruction) if not isinstance(instruction, list) else "<multimodal>",
            output=answer,
        )
        trajectory = Trajectory(
            uid=config.session_uid,
            name=self.name,
            task=task.id,
            steps=[step],
            output=answer,
        )
        return Episode(
            id=config.session_uid,
            task=task.id,
            trajectories=[trajectory],
        )


register_harness("simple", SimpleHarness)

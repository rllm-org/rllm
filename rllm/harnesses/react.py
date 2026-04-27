"""ReActHarness: one-shot LLM call for data tasks.

Default harness for catalog datasets (gsm8k, MATH, MMLU, etc.) where the
agent's "work" is a single chat completion. Sets ``trajectory.output``
to the LLM response so downstream score_fns can extract the answer.

Implements the rLLM ``AgentFlow`` protocol with no sandbox dependency.

For sandbox tasks (Harbor, SWE-bench), use :class:`rllm.tasks.harnesses.bash.BashHarness`
instead — it runs a multi-turn ReAct loop with bash tool calls inside the sandbox.
"""

from __future__ import annotations

import logging

from rllm.task import Task
from rllm.tasks.harness import register_harness
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)


_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer the question to the best of your ability."


class ReActHarness:
    """One-shot LLM harness for data tasks.

    No sandbox required. Pure ``AgentFlow`` — produces an Episode with
    a single Trajectory whose ``output`` is the LLM response.
    """

    name = "react"
    max_concurrent = 64

    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    def run(self, task: Task, config) -> Episode:
        from openai import OpenAI

        from rllm.eval.score_fns._resolver import get_verifier_system_prompt

        client = OpenAI(base_url=config.base_url, api_key="EMPTY")

        # Compose system prompt: base + verifier-specific output-format hint
        system_msg = self.system_prompt
        verifier_hint = get_verifier_system_prompt(task)
        if verifier_hint:
            system_msg = f"{system_msg}\n\n{verifier_hint}"

        instruction = task.instruction
        # Multimodal: pass through the content blocks as-is
        user_content = instruction if isinstance(instruction, list) else str(instruction)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        try:
            response = client.chat.completions.create(model=config.model, messages=messages)
            answer = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("ReActHarness LLM call failed for task %s: %s", task.id, e)
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


register_harness("react", ReActHarness)

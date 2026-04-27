"""ReActHarness: a host-side ReAct loop using the OpenAI SDK.

The harness runs in Python on the host machine and uses the sandbox as a
tool (executes shell commands via ``sandbox.exec``). It calls the LLM via
the LiteLLM proxy, so all LLM calls are visible to the gateway for trace
capture / training.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from rllm.tasks.harness import register_harness
from rllm.types import Step, Trajectory

if TYPE_CHECKING:
    from rllm.experimental.eval.types import AgentConfig
    from rllm.sandbox.protocol import Sandbox
    from rllm.tasks.task import Task

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are a skilled software engineer working inside a sandbox environment.
Complete the task by executing shell commands.

To run a command, wrap it in a ```bash code block like this:

```bash
echo 'Hello, world!' > hello.txt
```

After each command, you will see its output. \
When you are finished, respond with 'Task completed' (no code block)."""


_DONE_MARKERS = ("task completed", "task is complete", "done", "finished", "i have completed")


class ReActHarness:
    """Host-side ReAct loop. Default harness for tasks.

    Loop: prompt LLM → extract bash from response → exec in sandbox →
    feed output back → repeat until done or max_turns.
    """

    name = "react"

    def setup(self, sandbox: Sandbox, config: AgentConfig) -> None:  # noqa: D401
        """No-op for host-side harnesses."""

    def run(self, task: Task, sandbox: Sandbox, config: AgentConfig) -> Trajectory:
        from openai import OpenAI

        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        max_turns = task.rllm.max_turns or 50

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": task.instruction},
        ]
        steps: list[Step] = []

        for turn in range(max_turns):
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
            )
            assistant_msg = response.choices[0].message.content or ""
            logger.debug("[react turn %d] %d chars: %s", turn, len(assistant_msg), assistant_msg[:200])
            messages.append({"role": "assistant", "content": assistant_msg})

            steps.append(
                Step(
                    id=f"step-{turn}",
                    input=messages[-2]["content"] if turn == 0 else "",
                    output=assistant_msg,
                )
            )

            # Execute any bash command in the response (as agent_user if configured)
            command = _extract_command(assistant_msg)
            if command:
                try:
                    result = sandbox.exec(
                        command,
                        timeout=float(task.agent_timeout),
                        user=task.agent_user,
                    )
                except Exception as e:
                    result = f"Error: {e}"
                messages.append({"role": "user", "content": f"Command output:\n{result}"})

            # Done check (after executing any pending command)
            if _is_done(assistant_msg):
                break

            # No command to run and no done signal → agent stuck, stop
            if not command:
                break

        return Trajectory(
            uid=config.session_uid,
            name=self.name,
            task=task.name,
            steps=steps,
            output=steps[-1].output if steps else "",
        )


def _is_done(text: str) -> bool:
    """Heuristic: did the agent signal completion?"""
    lower = text.lower().strip()
    return any(lower.endswith(m) or lower.startswith(m) for m in _DONE_MARKERS)


def _extract_command(text: str) -> str | None:
    """Extract a bash command from a markdown code fence."""
    match = re.search(r"```(?:bash|shell|sh)\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


register_harness("react", ReActHarness)

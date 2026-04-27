"""ReActHarness: a host-side ReAct loop, implemented as a SandboxedAgentFlow.

Loop: prompt LLM → extract bash from response → exec in sandbox → feed
output back → repeat until done or ``[rllm].max_turns``. The harness
calls the LLM via the LiteLLM proxy at ``config.base_url`` so all LLM
calls are visible to the gateway for trace capture / training.

Conforms to the rLLM ``AgentFlow`` protocol — produces an Episode with
a single Trajectory.
"""

from __future__ import annotations

import logging
import re

from rllm.experimental.agents.sandboxed_agent import SandboxedAgentFlow
from rllm.task import Task
from rllm.tasks.harness import register_harness
from rllm.types import Episode, Step, Trajectory

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


class ReActHarness(SandboxedAgentFlow):
    """Default harness: a one-trajectory ReAct loop.

    The Runner sets the sandbox via ``set_sandbox()`` before ``run()`` is called.
    """

    name = "react"
    sandbox_backend = "docker"
    max_concurrent = 4

    def run(self, task: Task, config) -> Episode:
        from openai import OpenAI

        sandbox = self.sandbox
        if sandbox is None:
            raise RuntimeError("ReActHarness requires a sandbox. The Runner should set one before calling run().")

        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        max_turns = int(task.metadata.get("rllm", {}).get("max_turns") or 50)
        agent_timeout = float(task.metadata.get("agent_timeout", 600))
        agent_user = task.metadata.get("agent_user")

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": str(task.instruction)},
        ]
        steps: list[Step] = []

        for turn in range(max_turns):
            response = client.chat.completions.create(model=config.model, messages=messages)
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

            command = _extract_command(assistant_msg)
            if command:
                try:
                    result = sandbox.exec(command, timeout=agent_timeout, user=agent_user)
                except Exception as e:
                    result = f"Error: {e}"
                messages.append({"role": "user", "content": f"Command output:\n{result}"})

            if _is_done(assistant_msg):
                break
            if not command:
                break

        trajectory = Trajectory(
            uid=config.session_uid,
            name=self.name,
            task=task.id,
            steps=steps,
            output=steps[-1].output if steps else "",
        )
        return Episode(
            id=config.session_uid,
            task=task.id,
            trajectories=[trajectory],
        )


def _is_done(text: str) -> bool:
    lower = text.lower().strip()
    return any(lower.endswith(m) or lower.startswith(m) for m in _DONE_MARKERS)


def _extract_command(text: str) -> str | None:
    match = re.search(r"```(?:bash|shell|sh)\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


register_harness("react", ReActHarness)

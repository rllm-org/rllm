"""Terminal-Bench agent (stub).

Terminal-Bench tasks use Docker Compose for multi-container environments.
Full implementation will override setup_sandbox/teardown_sandbox
for custom container management via the terminal-bench package.

Requires: pip install terminal-agent[full]
"""

from __future__ import annotations

import logging

import openai

from rllm.experimental.agents.sandboxed_agent import SandboxedAgentFlow
from rllm.experimental.agents.tool_calling import ToolCallingMixin
from rllm.experimental.agents.tools.bash_tool import BashTool
from rllm.types import Episode, Trajectory

from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class TerminalAgentFlow(SandboxedAgentFlow, ToolCallingMixin):
    """Terminal-Bench agent stub.

    Uses ToolCallingMixin for multi-turn bash interactions.
    Full implementation will use terminal_bench.terminal.Terminal
    for Docker Compose environment management.
    """

    max_concurrent: int = 1
    sandbox_backend: str = "docker"

    def setup_sandbox(self, task, config):
        """Override: create Docker Compose env via terminal-bench package.

        TODO: Use terminal_bench.terminal.Terminal for Docker Compose
        management. For now, delegates to the default Docker sandbox.
        """
        super().setup_sandbox(task, config)

    def run(self, task: dict, config) -> Episode:
        instruction = task.get("instruction", task.get("question", ""))
        task_id = task.get("task_id", task.get("instance_id", "unknown"))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

        client = openai.OpenAI(base_url=config.base_url, api_key="not-needed")

        steps, messages, final_content = self.run_tool_loop(
            client=client,
            model=config.model,
            messages=messages,
            tools=[BashTool()],
            sandbox=self.sandbox,
            max_turns=50,
        )

        trajectory = Trajectory(
            name="terminal",
            task=task,
            steps=steps,
            output=final_content or "",
        )
        return Episode(
            id=f"{task_id}:0",
            task=task,
            trajectories=[trajectory],
            artifacts={"answer": final_content or ""},
        )


# Module-level singleton for plugin entry point
terminal_agent = TerminalAgentFlow()

"""ClaudeCodeHarness: runs the Claude Code CLI inside the sandbox.

Implemented as a SandboxedAgentFlow. Installs ``@anthropic-ai/claude-code``
(npm) inside the sandbox on first use (via ``on_sandbox_ready`` hook),
then invokes ``claude`` non-interactively. All LLM calls are routed
through the LiteLLM proxy via ``ANTHROPIC_BASE_URL`` so token-level
traces are captured by the gateway like host-side harnesses.
"""

from __future__ import annotations

import logging
import shlex

from rllm.experimental.agents.sandboxed_agent import SandboxedAgentFlow
from rllm.task import Task
from rllm.tasks.harness import register_harness
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)


_INSTALL_SCRIPT = r"""
set -e
if ! command -v claude >/dev/null 2>&1; then
    if ! command -v npm >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update -qq && apt-get install -y -qq nodejs npm
        elif command -v apk >/dev/null 2>&1; then
            apk add --no-cache nodejs npm
        else
            echo "Cannot install node/npm: no supported package manager" >&2
            exit 1
        fi
    fi
    npm install -g @anthropic-ai/claude-code
fi
"""


class ClaudeCodeHarness(SandboxedAgentFlow):
    """Run Anthropic's Claude Code CLI inside the sandbox."""

    name = "claude-code"
    sandbox_backend = "docker"
    max_concurrent = 4

    def on_sandbox_ready(self, task: dict, config) -> None:
        """Install claude-code if not already present (root)."""
        if self.sandbox is None:
            return
        try:
            self.sandbox.exec(_INSTALL_SCRIPT, timeout=600, user="root")
        except Exception as e:
            raise RuntimeError(f"Failed to install claude-code in sandbox: {e}") from e

    def run(self, task: Task, config) -> Episode:
        sandbox = self.sandbox
        if sandbox is None:
            raise RuntimeError("ClaudeCodeHarness requires a sandbox.")

        env = {
            "ANTHROPIC_BASE_URL": config.base_url,
            "ANTHROPIC_API_KEY": "sk-rllm-proxy",
            "ANTHROPIC_MODEL": config.model,
        }
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())

        instruction = str(task.instruction).strip()
        workdir = task.metadata.get("workdir", "/workspace")
        agent_timeout = float(task.metadata.get("agent_timeout", 600))
        agent_user = task.metadata.get("agent_user")

        cmd = f"cd {shlex.quote(workdir)} && {env_prefix} claude -p {shlex.quote(instruction)} --output-format text --permission-mode acceptEdits"

        try:
            output = sandbox.exec(cmd, timeout=agent_timeout, user=agent_user)
        except Exception as e:
            output = f"claude-code execution failed: {e}"
            logger.warning("ClaudeCodeHarness: %s", output)

        step = Step(id="step-0", input=instruction, output=output)
        trajectory = Trajectory(
            uid=config.session_uid,
            name=self.name,
            task=task.id,
            steps=[step],
            output=output,
        )
        return Episode(
            id=config.session_uid,
            task=task.id,
            trajectories=[trajectory],
        )


register_harness("claude-code", ClaudeCodeHarness)

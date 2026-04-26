"""ClaudeCodeHarness: runs the Claude Code CLI inside the sandbox.

The harness installs ``@anthropic-ai/claude-code`` (npm) inside the sandbox
on first use, then invokes ``claude`` non-interactively with the task
instruction.  All LLM calls are routed through the LiteLLM proxy via
``ANTHROPIC_BASE_URL``, so token-level traces are captured by the gateway
exactly like host-side harnesses.

Requires the sandbox to have ``node``/``npm`` available (or installable).
"""

from __future__ import annotations

import logging
import shlex
from typing import TYPE_CHECKING

from rllm.tasks.harness import register_harness
from rllm.types import Step, Trajectory

if TYPE_CHECKING:
    from rllm.experimental.eval.types import AgentConfig
    from rllm.sdk.sandbox.protocol import Sandbox
    from rllm.tasks.task import Task

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


class ClaudeCodeHarness:
    """Run Anthropic's Claude Code CLI inside the sandbox.

    The CLI is invoked in non-interactive mode (``-p``) with the task
    instruction. Set ``ANTHROPIC_BASE_URL`` so LLM traffic flows through
    the LiteLLM proxy.
    """

    name = "claude-code"

    def setup(self, sandbox: Sandbox, config: AgentConfig) -> None:
        """Install claude-code in the sandbox if not already present."""
        try:
            sandbox.exec(_INSTALL_SCRIPT, timeout=600)
        except Exception as e:
            raise RuntimeError(f"Failed to install claude-code in sandbox: {e}") from e

    def run(self, task: Task, sandbox: Sandbox, config: AgentConfig) -> Trajectory:
        # Route claude-code's API calls through the LiteLLM proxy. The proxy
        # exposes an Anthropic-compatible endpoint at /anthropic when running
        # in unified mode; tweak as needed for your deployment.
        env = {
            "ANTHROPIC_BASE_URL": config.base_url,
            "ANTHROPIC_API_KEY": "sk-rllm-proxy",  # any non-empty value; proxy ignores
            "ANTHROPIC_MODEL": config.model,
        }
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())

        instruction = task.instruction.strip()
        # Use -p for non-interactive print mode; --output-format text for plain stdout
        cmd = f"cd {shlex.quote(task.workdir)} && {env_prefix} claude -p {shlex.quote(instruction)} --output-format text --permission-mode acceptEdits"

        try:
            output = sandbox.exec(cmd, timeout=float(task.agent_timeout))
        except Exception as e:
            output = f"claude-code execution failed: {e}"
            logger.warning("ClaudeCodeHarness: %s", output)

        step = Step(id="step-0", input=instruction, output=output)
        return Trajectory(
            uid=config.session_uid,
            name=self.name,
            task=task.name,
            steps=[step],
            output=output,
        )


register_harness("claude-code", ClaudeCodeHarness)

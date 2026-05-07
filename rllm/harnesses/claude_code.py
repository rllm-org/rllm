"""ClaudeCodeHarness: runs the Claude Code CLI inside the sandbox.

Implemented as a SandboxedAgentFlow. Installs ``@anthropic-ai/claude-code``
(npm) inside the sandbox on first use (via ``on_sandbox_ready`` hook),
then invokes ``claude`` non-interactively. All LLM calls are routed
through the rLLM model gateway via ``ANTHROPIC_BASE_URL`` so wire-level
traces are captured by the gateway.

``run()`` returns ``None``; the gateway captures every LLM call and the
engine builds the trajectory during ``execute_tasks`` enrichment.
"""

from __future__ import annotations

import logging
import shlex

from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow
from rllm.types import AgentConfig, Task

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
        if self.sandbox is None:
            return
        try:
            self.sandbox.exec(_INSTALL_SCRIPT, timeout=600, user="root")
        except Exception as e:
            raise RuntimeError(f"Failed to install claude-code in sandbox: {e}") from e

    def run(self, task: Task, config: AgentConfig) -> None:
        sandbox = self.sandbox
        if sandbox is None:
            raise RuntimeError("ClaudeCodeHarness requires a sandbox.")

        # The gateway URL is already stamped with the per-session prefix
        # by the engine, so the CLI's calls hit /sessions/<uid>/v1/...
        # and the gateway tags every trace with this task's session.
        env = {
            "ANTHROPIC_BASE_URL": config.base_url.rstrip("/").removesuffix("/v1") or config.base_url,
            "ANTHROPIC_API_KEY": (config.metadata or {}).get("gateway_auth_token") or "sk-rllm-gateway",
            "ANTHROPIC_MODEL": config.model,
        }
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())

        instruction = str(task.instruction).strip()
        workdir = task.metadata.get("workdir")
        agent_timeout = float(task.metadata.get("agent_timeout", 600))
        agent_user = task.metadata.get("agent_user")

        cd_prefix = f"cd {shlex.quote(workdir)} && " if workdir else ""
        cmd = f"{cd_prefix}{env_prefix} claude -p {shlex.quote(instruction)} --output-format text --permission-mode acceptEdits"

        try:
            sandbox.exec(cmd, timeout=agent_timeout, user=agent_user)
        except Exception as e:
            logger.warning("ClaudeCodeHarness execution failed: %s", e)

        return None

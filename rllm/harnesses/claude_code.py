"""ClaudeCodeHarness: runs the Claude Code CLI inside the sandbox.

Subclass of :class:`BaseCliHarness`, so it inherits the install/run
lifecycle, gateway-trace-driven trajectory wiring (``run() -> None``),
host-loopback rewrite (so ``127.0.0.1`` resolves to ``host.docker.internal``
inside Docker), and bearer-token auth handling.

Anthropic's CLI reads ``ANTHROPIC_BASE_URL`` and ``ANTHROPIC_API_KEY``;
the gateway-routed URL is ``<base>/sessions/<uid>`` (the SDK appends
``/v1/messages`` itself so we strip a trailing ``/v1``).
"""

from __future__ import annotations

import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.types import AgentConfig, Task

# nvm-based bootstrap mirrors :mod:`rllm.harnesses.opencode`. apt-based
# nodejs/npm installs on swebench testbeds (Ubuntu jammy) routinely
# fail with stale GPG signatures; ``-qq`` masks the error and the
# script proceeds to ``npm`` which is then not on PATH.
_INSTALL_SCRIPT = r"""
set -e
export DEBIAN_FRONTEND=noninteractive
if ! command -v claude >/dev/null 2>&1; then
    if ! command -v node >/dev/null 2>&1; then
        if ! command -v curl >/dev/null 2>&1; then
            if command -v apt-get >/dev/null 2>&1; then
                apt-get update -qq && apt-get install -y -qq curl ca-certificates
            elif command -v apk >/dev/null 2>&1; then
                apk add --no-cache curl bash ca-certificates
            else
                echo "claude-code install requires curl" >&2
                exit 1
            fi
        fi
        export NVM_DIR="$HOME/.nvm"
        curl -fsSL -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
        \. "$NVM_DIR/nvm.sh"
        nvm install 22
    fi
    [ -s "$HOME/.nvm/nvm.sh" ] && \. "$HOME/.nvm/nvm.sh"
    npm install -g @anthropic-ai/claude-code
fi
"""


class ClaudeCodeHarness(BaseCliHarness):
    """Run Anthropic's Claude Code CLI inside the sandbox."""

    name = "claude-code"
    sandbox_backend = "docker"
    max_concurrent = 4
    stdout_log_path = "/tmp/claude-code.log"

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        # Anthropic's SDK appends ``/v1/messages`` itself, so trim a
        # trailing ``/v1`` from the gateway URL or it doubles up.
        gateway_url = self._container_url(config.base_url)
        anthropic_url = gateway_url.rstrip("/").removesuffix("/v1") or gateway_url
        return {
            "ANTHROPIC_BASE_URL": anthropic_url,
            "ANTHROPIC_API_KEY": self.gateway_api_key(config, "ANTHROPIC_API_KEY"),
            "ANTHROPIC_MODEL": config.model,
        }

    def build_invocation(
        self,
        instruction: str,
        task: Task,
        config: AgentConfig,
    ) -> str:
        # ``--bare`` is the key flag for non-interactive auth: claude
        # otherwise gates ``-p`` on an OAuth login check (``Not logged
        # in · Please run /login``) that exits 0 with no trace, even
        # when ``ANTHROPIC_API_KEY`` is set. Bare mode skips OAuth +
        # keychain reads and uses the env-var key directly, which is
        # what we want when routing through the rllm gateway.
        # Source nvm so the npm-installed ``claude`` binary is on PATH;
        # silent no-op when node came from a system package.
        return (
            f"{self._cd_prefix(task)}"
            f". $HOME/.nvm/nvm.sh 2>/dev/null; "
            f"claude --bare -p {shlex.quote(instruction)} "
            f"--output-format text --permission-mode acceptEdits "
            f"</dev/null 2>&1 | tee {shlex.quote(self.stdout_log_path)}"
        )

"""QwenCodeHarness: runs Alibaba's Qwen Code CLI inside the sandbox.

Qwen Code (``@qwen-code/qwen-code``) is a fork of gemini-cli wired for
Qwen models over the OpenAI-compatible API. It reads
``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` / ``OPENAI_MODEL`` natively —
no config file is needed when those three are set, which is the same
shape as claude_code.py (env-only).

``run()`` returns ``None``; the gateway captures every LLM call and
the engine builds the trajectory.
"""

from __future__ import annotations

import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.types import AgentConfig, Task

# apt branch hardened against arm64 ubuntu-ports GPG flakes — same
# rationale as claude_code.py's install script.
_INSTALL_SCRIPT = r"""
set -e
export DEBIAN_FRONTEND=noninteractive
if ! command -v qwen >/dev/null 2>&1; then
    if ! command -v curl >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update -qq \
                -o Acquire::AllowInsecureRepositories=true \
                -o Acquire::AllowDowngradeToInsecureRepositories=true \
                -o Acquire::Check-Valid-Until=false 2>/dev/null || true
            apt-get install -y -qq --no-install-recommends --allow-unauthenticated \
                -o Acquire::AllowInsecureRepositories=true \
                curl ca-certificates
        elif command -v apk >/dev/null 2>&1; then
            apk add --no-cache curl bash ca-certificates
        fi
    fi
    command -v curl >/dev/null 2>&1 \
        || { echo "qwen-code install: failed to bootstrap curl in sandbox" >&2; exit 1; }
    if ! command -v node >/dev/null 2>&1; then
        export NVM_DIR="$HOME/.nvm"
        curl -fsSL -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
        \. "$NVM_DIR/nvm.sh"
        nvm install 22
    fi
    [ -s "$HOME/.nvm/nvm.sh" ] && \. "$HOME/.nvm/nvm.sh"
    npm install -g @qwen-code/qwen-code
fi
qwen --version >/dev/null
"""


class QwenCodeHarness(BaseCliHarness):
    """Run Qwen Code inside the sandbox."""

    name = "qwen-code"
    sandbox_backend = "docker"
    max_concurrent = 4
    stdout_log_path = "/tmp/qwen-code.log"

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        gateway_url = config.base_url
        # Qwen Code uses the OpenAI-compatible client, so a bare model
        # id (the part after ``provider/``) is what the wire expects;
        # the gateway routes by model name, not by env var.
        _, model_id, _ = self.ensure_provider_prefix(config.model)
        return {
            "OPENAI_API_KEY": self.gateway_api_key(config, "OPENAI_API_KEY"),
            "OPENAI_BASE_URL": gateway_url,
            "OPENAI_MODEL": model_id,
        }

    def build_invocation(
        self,
        instruction: str,
        task: Task,
        config: AgentConfig,
    ) -> str:
        # ``-p`` (alias of ``--prompt``) is qwen-code's non-interactive
        # mode — the CLI runs the agent loop and exits when the model
        # signals done. ``--yolo`` auto-approves every tool call (matches
        # the unattended-eval contract; we're inside the rLLM sandbox
        # already so per-call approval gates add nothing).
        # ``</dev/null`` keeps Modal-style execs from blocking on a
        # never-closing stdin — same lesson learned in opencode.py.
        _, model_id, _ = self.ensure_provider_prefix(config.model)
        return (
            f"{self._cd_prefix(task)}"
            f". $HOME/.nvm/nvm.sh 2>/dev/null; "
            f"qwen --yolo --model {shlex.quote(model_id)} "
            f"-p {shlex.quote(instruction)} "
            f"</dev/null 2>&1 | tee {shlex.quote(self.stdout_log_path)}"
        )

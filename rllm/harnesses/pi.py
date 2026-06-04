"""PiHarness: runs Mario Zechner's pi-coding-agent CLI inside the sandbox.

pi-coding-agent (``@mariozechner/pi-coding-agent``) is an npm-distributed
coding agent that picks its model from ``--provider`` + ``--model`` CLI
flags. Under the hood it uses the Vercel ai-sdk, which honours
``OPENAI_BASE_URL`` for the openai provider and ``ANTHROPIC_BASE_URL``
for anthropic — so the gateway routes by env var the same way the other
ai-sdk-based harnesses do.

``--print --mode json --no-session`` is the non-interactive contract:
emit a single JSON stream of events and exit instead of keeping a
persisted session open.

``run()`` returns ``None``; the gateway captures every LLM call and
the engine builds the trajectory.
"""

from __future__ import annotations

import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.types import AgentConfig, Task

_INSTALL_SCRIPT = r"""
set -e
export DEBIAN_FRONTEND=noninteractive
if ! command -v pi >/dev/null 2>&1; then
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
        || { echo "pi install: failed to bootstrap curl in sandbox" >&2; exit 1; }
    if ! command -v node >/dev/null 2>&1; then
        export NVM_DIR="$HOME/.nvm"
        curl -fsSL -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
        \. "$NVM_DIR/nvm.sh"
        nvm install 22
    fi
    [ -s "$HOME/.nvm/nvm.sh" ] && \. "$HOME/.nvm/nvm.sh"
    npm install -g @mariozechner/pi-coding-agent
fi
"""

# Provider names ``pi`` accepts as ``--provider``. We pick whichever one
# matches the model's likely upstream (gateway routes regardless).
_PI_PROVIDER_MAP = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "groq": "groq",
    "mistral": "mistral",
    "xai": "xai",
    "openrouter": "openrouter",
    "huggingface": "huggingface",
    "deepseek": "openai",  # pi has no deepseek provider; route as openai-compatible
}


class PiHarness(BaseCliHarness):
    """Run Mario Zechner's pi-coding-agent inside the sandbox."""

    name = "pi"
    sandbox_backend = "docker"
    max_concurrent = 4
    stdout_log_path = "/tmp/pi.log"

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def _pi_provider(self, model: str) -> tuple[str, str]:
        """Return ``(pi_provider, model_id)`` for *model*."""
        provider, model_id, _ = self.ensure_provider_prefix(model)
        return _PI_PROVIDER_MAP.get(provider, "openai"), model_id

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        gateway_url = self._container_url(config.base_url)
        api_key = self.gateway_api_key(config, "OPENAI_API_KEY")
        # Set both the openai-style and anthropic-style env vars so the
        # ai-sdk providers route through the gateway regardless of which
        # ``--provider`` we pick on the CLI.
        return {
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": gateway_url,
            "ANTHROPIC_API_KEY": api_key,
            "ANTHROPIC_BASE_URL": gateway_url.rstrip("/").removesuffix("/v1") or gateway_url,
        }

    def build_invocation(
        self,
        instruction: str,
        task: Task,
        config: AgentConfig,
    ) -> str:
        # ``--print --mode json --no-session`` — single-shot, structured
        # output, no persisted session. The prompt is the trailing
        # positional argument.
        pi_provider, model_id = self._pi_provider(config.model)
        return (
            f"{self._cd_prefix(task)}"
            f". $HOME/.nvm/nvm.sh 2>/dev/null; "
            f"pi --print --mode json --no-session "
            f"--provider {shlex.quote(pi_provider)} "
            f"--model {shlex.quote(model_id)} "
            f"{shlex.quote(instruction)} "
            f"</dev/null 2>&1 | tee {shlex.quote(self.stdout_log_path)}"
        )

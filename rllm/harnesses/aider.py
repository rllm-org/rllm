"""AiderHarness: runs Paul Gauthier's aider CLI inside the sandbox.

aider uses litellm under the hood, so it honors ``OPENAI_API_BASE`` /
``OPENAI_BASE_URL`` / ``ANTHROPIC_BASE_URL`` for routing. ``--yes``
auto-confirms every prompt (including git operations and shell
suggestions) so the run terminates without user input. The ``--message``
flag is aider's non-interactive equivalent of stdin chat — the CLI
runs one turn, applies edits, and exits.

``run()`` returns ``None``; the gateway captures every LLM call and
the engine builds the trajectory.
"""

from __future__ import annotations

import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.types import AgentConfig, Task

# Provider → litellm-style API key env var. aider reads ``--api-key
# <provider>=<key>`` form OR the litellm default env vars.
_PROVIDER_KEY_VAR = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
}

# apt branch hardened against arm64 ubuntu-ports GPG flakes (see
# claude_code.py for the longer explanation).
_INSTALL_SCRIPT = r"""
set -e
export DEBIAN_FRONTEND=noninteractive
if ! { export PATH="$HOME/.local/bin:$PATH"; command -v aider >/dev/null 2>&1; }; then
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
        || { echo "aider install: failed to bootstrap curl in sandbox" >&2; exit 1; }
    # aider's official installer drops the binary into ``$HOME/.local/bin``
    # via uv; matches the curl-install pattern used by claude-code.
    curl -LsSf https://aider.chat/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
aider --version >/dev/null
"""


class AiderHarness(BaseCliHarness):
    """Run Paul Gauthier's aider CLI inside the sandbox."""

    name = "aider"
    sandbox_backend = "docker"
    max_concurrent = 4
    stdout_log_path = "/tmp/aider.log"

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        gateway_url = config.base_url
        provider, _, _ = self.ensure_provider_prefix(config.model)
        api_key = self.gateway_api_key(config, _PROVIDER_KEY_VAR.get(provider, "OPENAI_API_KEY"))

        env: dict[str, str] = {
            # litellm reads OPENAI_API_BASE; aider also reads OPENAI_BASE_URL.
            # Both are set for safety against version drift.
            "OPENAI_API_BASE": gateway_url,
            "OPENAI_BASE_URL": gateway_url,
            # Anthropic SDK appends ``/v1/messages`` itself — strip a
            # trailing ``/v1`` or the URL doubles to ``/v1/v1/messages``.
            "ANTHROPIC_BASE_URL": gateway_url.rstrip("/").removesuffix("/v1") or gateway_url,
            # Forward the provider key litellm will look up. When the
            # gateway requires inbound auth, this becomes the bearer token.
            _PROVIDER_KEY_VAR.get(provider, "OPENAI_API_KEY"): api_key,
            # aider's built-in analytics POST adds latency and leaks the
            # gateway URL to mixpanel — disable.
            "AIDER_ANALYTICS": "false",
        }
        return env

    def build_invocation(
        self,
        instruction: str,
        task: Task,
        config: AgentConfig,
    ) -> str:
        # ``--yes`` auto-confirms every diff/shell prompt. ``--no-stream``
        # so the gateway sees a single response instead of one stream
        # chunk per token (cleaner traces). ``--no-git`` because Harbor
        # tasks aren't always git repos and aider otherwise refuses to
        # start. ``--message`` is the non-interactive single-turn flag.
        _, _, qualified = self.ensure_provider_prefix(config.model)
        return (
            f"{self._cd_prefix(task)}"
            f'export PATH="$HOME/.local/bin:$PATH"; '
            f"aider --yes --no-stream --no-git "
            f"--model={shlex.quote(qualified)} "
            f"--message={shlex.quote(instruction)} "
            f"</dev/null 2>&1 | tee {shlex.quote(self.stdout_log_path)}"
        )

"""OpenCodeHarness: runs the opencode-ai CLI inside the sandbox.

opencode reads ``OPENAI_BASE_URL`` from env *and* requires the same
URL be present in ``~/.config/opencode/opencode.json`` as
``provider.<name>.options.baseURL`` — env var alone is not honored
when the provider is registered via config.
"""

from __future__ import annotations

import json
import os
import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.types import AgentConfig, Task

# Map ``provider`` (left side of provider/model) to (env var holding the
# user's real API key, opencode provider config name). Most providers
# work with just the env-var-and-config-file combination; add to this
# table as new ones come up.
_PROVIDER_AUTH = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
}

_INSTALL_SCRIPT = r"""
set -e
if ! command -v opencode >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -y -qq curl ca-certificates
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache curl bash ca-certificates
    fi
    if ! command -v node >/dev/null 2>&1; then
        export NVM_DIR="$HOME/.nvm"
        curl -fsSL -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
        \. "$NVM_DIR/nvm.sh"
        nvm install 22
    fi
    npm install -g opencode-ai@latest
fi
opencode --version >/dev/null
"""


class OpenCodeHarness(BaseCliHarness):
    """Run opencode-ai inside the sandbox."""

    name = "opencode"
    sandbox_backend = "docker"
    max_concurrent = 4
    stdout_log_path = "/tmp/opencode.log"

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def _split_provider(self, model: str) -> tuple[str, str, str]:
        """Return ``(provider, model_id, qualified)`` for *model*.

        Bare names (``gpt-4o``) get a provider inferred from the name
        pattern so rllm setup's configured model name works without the
        user having to add ``openai/`` / ``anthropic/`` themselves.
        """
        return self.ensure_provider_prefix(model)

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        provider, _, _ = self._split_provider(config.model)
        gateway_url = self._container_url(config.base_url)

        env: dict[str, str] = {
            # opencode uses an OpenAI-shaped client for openai/anthropic
            # alike; the gateway routes by model-name regardless.
            "OPENAI_BASE_URL": gateway_url,
            "OPENAI_API_BASE": gateway_url,
            "ANTHROPIC_BASE_URL": gateway_url.rstrip("/").removesuffix("/v1") or gateway_url,
            "OPENCODE_FAKE_VCS": "git",
        }

        # Forward the gateway bearer token (when public URL → public auth)
        # or the user's actual provider key, or a placeholder. The gateway
        # re-stamps auth with the real upstream key before forwarding.
        api_key_var = _PROVIDER_AUTH.get(provider, "OPENAI_API_KEY")
        env[api_key_var] = self.gateway_api_key(config, api_key_var)
        return env

    # Provider name we register the gateway under inside opencode.json.
    # Has to be NEW (not "openai" / "anthropic"), because opencode treats
    # those names as known providers and validates the model id against
    # its bundled ``models.dev`` registry — any custom name (e.g.
    # ``gpt-5.4-mini`` from ``rllm setup``) raises
    # ``ProviderModelNotFoundError`` even when listed in our ``models``
    # block. A custom-named provider with ``npm: @ai-sdk/openai-compatible``
    # bypasses that validation.
    _RLLM_PROVIDER_ID = "rllm-gateway"

    def write_configs(
        self,
        task: Task,
        config: AgentConfig,
        env: dict[str, str],
    ) -> None:
        _, model_id, _ = self._split_provider(config.model)
        gateway_url = self._container_url(config.base_url)
        api_key_var = _PROVIDER_AUTH.get(self._split_provider(config.model)[0], "OPENAI_API_KEY")
        # Same key build_env injected — bearer token when gateway is
        # exposed, else the user's real provider key. Reads from env
        # (already populated by build_env) for consistency.
        api_key = env.get(api_key_var, self.gateway_api_key(config, api_key_var))

        # ``npm: "@ai-sdk/openai-compatible"`` tells opencode to treat this
        # as a generic OpenAI-shaped endpoint. Model ids under this provider
        # are accepted as-is (no models.dev lookup), so arbitrary names
        # like ``gpt-5.4-mini`` work end-to-end.
        opencode_config = {
            "provider": {
                self._RLLM_PROVIDER_ID: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "rLLM Gateway",
                    "options": {
                        "baseURL": gateway_url,
                        "apiKey": api_key,
                    },
                    "models": {model_id: {}},
                }
            }
        }
        content = json.dumps(opencode_config, indent=2)
        # Write via a heredoc whose target path is *unquoted* in the shell
        # so ``$HOME`` expands at runtime. ``_heredoc_write`` shlex-quotes
        # its target, which silently turns ``$HOME/.config/opencode/...``
        # into a literal directory named ``$HOME`` under the cwd — opencode
        # then sees no config and falls back to its built-in registry,
        # giving the misleading ``ProviderModelNotFoundError``.
        marker = f"_RLLM_OPENCODE_EOF_{os.urandom(4).hex()}"
        cmd = f"mkdir -p $HOME/.config/opencode && cat > $HOME/.config/opencode/opencode.json << '{marker}'\n{content}\n{marker}"
        self._exec_agent(cmd, env=env)

    def build_invocation(
        self,
        instruction: str,
        task: Task,
        config: AgentConfig,
    ) -> str:
        # Use the custom-provider id we registered, not the inferred one.
        # opencode looks the model up under whatever provider prefix the
        # ``--model`` flag carries.
        _, model_id, _ = self._split_provider(config.model)
        qualified = f"{self._RLLM_PROVIDER_ID}/{model_id}"
        return (
            f"{self._cd_prefix(task)}"
            f". $HOME/.nvm/nvm.sh 2>/dev/null; "
            f"opencode --model={shlex.quote(qualified)} run "
            f"--format=json --dangerously-skip-permissions "
            f"-- {shlex.quote(instruction)} "
            f"2>&1 | tee {shlex.quote(self.stdout_log_path)}"
        )

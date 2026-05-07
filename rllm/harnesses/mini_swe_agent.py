"""MiniSweAgentHarness: runs the mini-swe-agent CLI inside the sandbox.

mini-swe-agent uses litellm under the hood, so it picks up
``OPENAI_API_BASE`` for OpenAI-shaped backends and a model-derived
provider key (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` / …). The
gateway routes by model name regardless.

``run()`` returns ``None``; the gateway captures every LLM call and
the engine builds the trajectory.
"""

from __future__ import annotations

import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.types import AgentConfig, Task

# Model-name prefix → provider env-var mapping. Mirrors litellm's logic
# without taking the dependency.
_PROVIDER_KEYS = (
    ("anthropic/", "ANTHROPIC_API_KEY"),
    ("claude", "ANTHROPIC_API_KEY"),  # bare claude-* model names
    ("openai/", "OPENAI_API_KEY"),
    ("gpt-", "OPENAI_API_KEY"),
    ("o1", "OPENAI_API_KEY"),
    ("deepseek/", "DEEPSEEK_API_KEY"),
    ("groq/", "GROQ_API_KEY"),
)


def _provider_key_var(model: str) -> str:
    name = model.lower()
    for prefix, var in _PROVIDER_KEYS:
        if name.startswith(prefix) or prefix in name:
            return var
    return "OPENAI_API_KEY"


_INSTALL_SCRIPT = r"""
set -e
# DEBIAN_FRONTEND=noninteractive is mandatory: ``apt-get install python3``
# pulls in ``tzdata``, which triggers a debconf timezone prompt and
# hangs forever on Modal sandboxes (Docker exec falls back to
# Noninteractive automatically when there's no TTY; Modal does not).
export DEBIAN_FRONTEND=noninteractive
if ! command -v mini-swe-agent >/dev/null 2>&1; then
    # Only fall back to apt when curl is missing — swebench testbeds
    # have expired Ubuntu jammy repo signatures, so unconditional
    # ``apt-get update`` fails with GPG errors and stops the script
    # before uv even gets a chance.
    if ! command -v curl >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update -qq && apt-get install -y -qq curl ca-certificates git
        elif command -v apk >/dev/null 2>&1; then
            apk add --no-cache curl bash ca-certificates git
        fi
    fi
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
    uv tool install mini-swe-agent
fi
"""


class MiniSweAgentHarness(BaseCliHarness):
    """Run mini-swe-agent inside the sandbox."""

    name = "mini-swe-agent"
    sandbox_backend = "docker"
    max_concurrent = 4
    stdout_log_path = "/tmp/mini-swe-agent.log"

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        gateway_url = self._container_url(config.base_url)
        env: dict[str, str] = {
            # Legacy v1 wizard-skip (still honoured by some forks); v2
            # ignores it and instead checks for ~/.config/mini-swe-agent/.env
            # which we write from ``write_configs``.
            "MSWEA_CONFIGURED": "true",
            # Don't fail when the gateway-routed model isn't in litellm's cost table.
            "MSWEA_COST_TRACKING": "ignore_errors",
            "OPENAI_API_BASE": gateway_url,
            "OPENAI_BASE_URL": gateway_url,
            "ANTHROPIC_BASE_URL": gateway_url.rstrip("/").removesuffix("/v1") or gateway_url,
        }

        # Forward the provider key litellm will look up. When the
        # gateway requires inbound auth, this becomes the bearer token
        # (gateway re-stamps with the real upstream key).
        api_var = _provider_key_var(config.model)
        env[api_var] = self.gateway_api_key(config, api_var)
        return env

    def write_configs(
        self,
        task: Task,
        config: AgentConfig,
        env: dict[str, str],
    ) -> None:
        """Write ``~/.config/mini-swe-agent/.env`` so mini-swe-agent v2 skips the setup wizard.

        v2's wizard fires whenever this file is missing — even with
        ``MSWEA_CONFIGURED=true`` in the environment. Pre-seeding it
        with the model + provider key is the only reliable bypass
        observed across versions ≥ 2.2.
        """
        _, _, qualified = self.ensure_provider_prefix(config.model)
        api_var = _provider_key_var(config.model)
        api_key = env.get(api_var, self.gateway_api_key(config, api_var))

        # Dotenv lines mini-swe-agent v2 reads on startup. The base
        # URL must live HERE (not just in process env) because v2 loads
        # the dotenv with ``override=True`` — it would otherwise unset
        # ``OPENAI_API_BASE`` we exported in :meth:`build_env`, sending
        # every call to api.openai.com and bypassing the gateway.
        gateway_url = self._container_url(config.base_url)
        dotenv_lines = [
            f"MSWEA_GLOBAL_MODEL={qualified}",
            f"{api_var}={api_key}",
            f"OPENAI_API_BASE={gateway_url}",
            f"OPENAI_BASE_URL={gateway_url}",
            f"ANTHROPIC_BASE_URL={gateway_url.rstrip('/').removesuffix('/v1') or gateway_url}",
            "MSWEA_CONFIGURED=true",
            "MSWEA_COST_TRACKING=ignore_errors",
        ]
        content = "\n".join(dotenv_lines)
        path = "$HOME/.config/mini-swe-agent/.env"
        # ``_heredoc_write`` quotes the target path, which kills
        # ``$HOME`` expansion — write the heredoc inline instead.
        self._exec_agent(
            f"mkdir -p $HOME/.config/mini-swe-agent && cat > {path} << 'MSWEA_DOTENV_EOF'\n{content}\nMSWEA_DOTENV_EOF",
            env=env,
        )

    def build_invocation(
        self,
        instruction: str,
        task: Task,
        config: AgentConfig,
    ) -> str:
        # mini-swe-agent insists on ``provider/model``; infer the prefix
        # when the user passed a bare name from rllm setup.
        _, _, qualified = self.ensure_provider_prefix(config.model)

        # NOTE: gateway routing relies on ``OPENAI_API_BASE`` in the
        # process environment. ``-c key=value`` overrides on the CLI
        # are NOT layered on top of mini.yaml in v2 — they replace it,
        # which breaks the build with missing ``system_template`` etc.
        # The dotenv we write in :meth:`write_configs` carries the base
        # URL into the agent's environment so litellm picks it up.
        return (
            f"{self._cd_prefix(task)}"
            f'export PATH="$HOME/.local/bin:$PATH"; '
            f"mini-swe-agent --yolo "
            f"--model={shlex.quote(qualified)} "
            f"--task={shlex.quote(instruction)} "
            f"--exit-immediately "
            f"2>&1 | tee {shlex.quote(self.stdout_log_path)}"
        )

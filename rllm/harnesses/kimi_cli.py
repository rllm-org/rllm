"""KimiCliHarness: runs Moonshot's kimi-cli inside the sandbox.

kimi-cli has its own config-driven routing model: providers are
declared in ``~/.kimi/config.json`` (or a path passed via
``--config-file``) and each model selects one of those providers.
We register the rLLM gateway as a custom OpenAI-compatible provider
and point our model at it — mirrors how :mod:`rllm.harnesses.opencode`
handles a similar config-first agent.

Two non-obvious facts:

1. kimi reads prompts over a JSON-RPC ``wire`` protocol on stdin, not
   from argv or a flag. Harbor opens stdin with ``(echo <json>; sleep
   86400)`` to keep the pipe alive while the agent streams responses
   back. ``--yolo`` auto-approves every tool call.
2. kimi accepts arbitrary ``provider.type`` values; ``openai_legacy``
   is the wire-compatible setting for any plain ``/v1/chat/completions``
   backend — which is what the gateway exposes.

``run()`` returns ``None``; the gateway captures every LLM call and
the engine builds the trajectory.
"""

from __future__ import annotations

import json
import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.types import AgentConfig, Task

_INSTALL_SCRIPT = r"""
set -e
export DEBIAN_FRONTEND=noninteractive
if ! { export PATH="$HOME/.local/bin:$PATH"; command -v kimi >/dev/null 2>&1; }; then
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
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | bash
    fi
    export PATH="$HOME/.local/bin:$PATH"
    uv tool install --python 3.13 kimi-cli
fi
export PATH="$HOME/.local/bin:$PATH"
kimi --version >/dev/null
"""

_KIMI_CONFIG_PATH = "/tmp/kimi-config.json"
_RLLM_KIMI_PROVIDER = "rllm-gateway"


class KimiCliHarness(BaseCliHarness):
    """Run Moonshot's kimi-cli inside the sandbox."""

    name = "kimi-cli"
    sandbox_backend = "docker"
    max_concurrent = 4
    stdout_log_path = "/tmp/kimi-cli.log"

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        # Keep PATH aware of $HOME/.local/bin (where ``uv tool install``
        # drops the kimi shim). Auth lives in the JSON config we write
        # below — these env vars are present as a fallback for code
        # paths inside kimi-cli that read them directly.
        api_key = self.gateway_api_key(config, "OPENAI_API_KEY")
        return {
            "OPENAI_API_KEY": api_key,
            "MOONSHOT_API_KEY": api_key,
            "KIMI_API_KEY": api_key,
        }

    def write_configs(
        self,
        task: Task,
        config: AgentConfig,
        env: dict[str, str],
    ) -> None:
        """Write kimi-cli's config JSON pointing at the gateway as a custom provider."""
        gateway_url = self._container_url(config.base_url)
        _, model_id, _ = self.ensure_provider_prefix(config.model)
        api_key = env.get("OPENAI_API_KEY", self.gateway_api_key(config, "OPENAI_API_KEY"))

        # ``type=openai_legacy`` is kimi's name for a plain OpenAI-
        # compatible ``/v1/chat/completions`` endpoint, which is what
        # the gateway exposes. max_context_size is large enough for
        # any current frontier model; the gateway forwards as-is.
        kimi_config = {
            "default_model": model_id,
            "default_yolo": True,
            "providers": {
                _RLLM_KIMI_PROVIDER: {
                    "type": "openai_legacy",
                    "base_url": gateway_url,
                    "api_key": api_key,
                },
            },
            "models": {
                model_id: {
                    "provider": _RLLM_KIMI_PROVIDER,
                    "model": model_id,
                    "max_context_size": 200000,
                },
            },
        }
        content = json.dumps(kimi_config, indent=2)
        self._exec_agent(self._heredoc_write(_KIMI_CONFIG_PATH, content), env=env)

    def build_invocation(
        self,
        instruction: str,
        task: Task,
        config: AgentConfig,
    ) -> str:
        # kimi reads the prompt from stdin as a JSON-RPC ``prompt``
        # method. ``--wire`` enables that protocol; ``--yolo`` auto-
        # approves every tool call. ``sleep 86400`` keeps the pipe open
        # so kimi can stream responses back (it exits when the model
        # signals completion).
        wire_request = json.dumps({"jsonrpc": "2.0", "method": "prompt", "params": {"text": instruction}, "id": 1})
        return (
            f"{self._cd_prefix(task)}"
            f'export PATH="$HOME/.local/bin:$PATH"; '
            f"(echo {shlex.quote(wire_request)}; sleep 86400) "
            f"| kimi --config-file {shlex.quote(_KIMI_CONFIG_PATH)} --wire --yolo "
            f"2>&1 | tee {shlex.quote(self.stdout_log_path)}"
        )

"""OpenHandsHarness: runs All Hands AI's OpenHands inside the sandbox.

OpenHands (formerly OpenDevin) is installed into a dedicated uv-managed
Python venv at ``/opt/openhands-venv`` and invoked as a Python module
(``python -m openhands.core.main``). The CLI reads its model + base URL
+ API key from ``LLM_MODEL`` / ``LLM_BASE_URL`` / ``LLM_API_KEY`` env
vars (litellm-style under the hood, so OpenAI/Anthropic-compatible
endpoints both work — the gateway routes by model name).

Sandbox-mode env: ``RUNTIME=local`` keeps the agent's tool execution
on the same machine instead of spinning up an inner docker container;
``RUN_AS_OPENHANDS=false`` + ``SU_TO_USER=false`` avoid user-switching
on systems without the ``openhands`` user account.

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
if [ ! -x /opt/openhands-venv/bin/python ]; then
    # openhands needs git + a working build toolchain for some optional
    # native deps (sandboxed shell uses ptyprocess, etc.).
    if ! command -v curl >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update -qq \
                -o Acquire::AllowInsecureRepositories=true \
                -o Acquire::AllowDowngradeToInsecureRepositories=true \
                -o Acquire::Check-Valid-Until=false 2>/dev/null || true
            apt-get install -y -qq --no-install-recommends --allow-unauthenticated \
                -o Acquire::AllowInsecureRepositories=true \
                curl ca-certificates git build-essential tmux
        elif command -v apk >/dev/null 2>&1; then
            apk add --no-cache curl bash ca-certificates git build-base tmux
        fi
    fi
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
    uv python install 3.12
    uv venv /opt/openhands-venv --python 3.12
    # Pin to the last pre-1.0 release: openhands-ai 1.x is a ground-up
    # rewrite that removes ``openhands.core.main`` (the single-shot CLI
    # entrypoint this harness invokes). The 1.x API only exposes a
    # long-running ``agent-server`` HTTP service, which needs a separate
    # client integration. Pre-1.0 keeps working with ``--task=<prompt>``.
    # uv handles installs without pip — calling ``python -m pip`` on a
    # bare ``uv venv`` fails with ``No module named pip``.
    uv pip install --python /opt/openhands-venv/bin/python "openhands-ai<1"
fi
/opt/openhands-venv/bin/python -c "import openhands" >/dev/null
"""


class OpenHandsHarness(BaseCliHarness):
    """Run All Hands AI's OpenHands inside the sandbox."""

    name = "openhands"
    sandbox_backend = "docker"
    max_concurrent = 4
    stdout_log_path = "/tmp/openhands.log"
    # Heavier install (uv + python toolchain + openhands-ai), allow more time.
    install_timeout = 1200

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        gateway_url = self._container_url(config.base_url)
        api_key = self.gateway_api_key(config, "OPENAI_API_KEY")
        _, _, qualified = self.ensure_provider_prefix(config.model)

        return {
            # Primary routing trio — openhands' LLM config reads these.
            "LLM_MODEL": qualified,
            "LLM_BASE_URL": gateway_url,
            "LLM_API_KEY": api_key,
            # Belt-and-braces: a few openhands code paths inspect the
            # generic openai/anthropic vars directly (e.g. for the
            # condenser / summariser sub-agent).
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": gateway_url,
            "ANTHROPIC_API_KEY": api_key,
            "ANTHROPIC_BASE_URL": gateway_url.rstrip("/").removesuffix("/v1") or gateway_url,
            # Sandbox-mode flags: keep agent tools running in-process
            # rather than spinning a nested docker container, and skip
            # the user-switching dance that assumes an ``openhands`` user.
            "RUNTIME": "local",
            "RUN_AS_OPENHANDS": "false",
            "SU_TO_USER": "false",
            # Prompt-extension prefixes can confuse the model when the
            # gateway is routing to a non-Anthropic backend.
            "AGENT_ENABLE_PROMPT_EXTENSIONS": "false",
        }

    def build_invocation(
        self,
        instruction: str,
        task: Task,
        config: AgentConfig,
    ) -> str:
        # Invoke the venv's python directly so PATH activation isn't required.
        # ``--task`` is openhands' single-shot entrypoint — runs the agent
        # loop until the model signals completion or hits the turn cap.
        return f"{self._cd_prefix(task)}/opt/openhands-venv/bin/python -m openhands.core.main --task={shlex.quote(instruction)} </dev/null 2>&1 | tee {shlex.quote(self.stdout_log_path)}"

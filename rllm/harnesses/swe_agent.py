"""SweAgentHarness: runs Princeton's SWE-agent inside the sandbox.

SWE-agent is the heavier ancestor of mini-swe-agent — installed from
git into a dedicated uv-managed venv at ``/opt/sweagent-venv``, invoked
as ``sweagent run`` with a YAML config and a problem-statement file.
It uses litellm internally, so ``OPENAI_BASE_URL`` / ``ANTHROPIC_BASE_URL``
route requests through the gateway.

The harness writes the task instruction to a file because
``--problem_statement.path`` only accepts a file, not a literal.

``run()`` returns ``None``; the gateway captures every LLM call and
the engine builds the trajectory.

NOTE: harbor's swe-agent agent enforces root-user execution. We rely on
``BaseCliHarness._exec_root`` for install (same as every other harness)
and let the run step inherit whatever user the task image declares —
swe-agent's ``--env.deployment.type=local`` mode is more permissive
than the docker-in-docker mode it ships with by default.
"""

from __future__ import annotations

import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.types import AgentConfig, Task

_INSTALL_SCRIPT = r"""
set -e
export DEBIAN_FRONTEND=noninteractive
if [ ! -x /opt/sweagent-venv/bin/sweagent ]; then
    # ``python3`` is non-optional: SWE-agent's internal state-collector
    # tool shells out to ``/usr/bin/env python3`` between turns, and on
    # minimal images (e.g. ubuntu:24.04) it's absent — every tool
    # invocation then fails with exit 127 and the agent loops forever
    # retrying. Install eagerly even when curl+git are already present.
    if ! command -v curl >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 \
            || ! command -v python3 >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update -qq \
                -o Acquire::AllowInsecureRepositories=true \
                -o Acquire::AllowDowngradeToInsecureRepositories=true \
                -o Acquire::Check-Valid-Until=false 2>/dev/null || true
            apt-get install -y -qq --no-install-recommends --allow-unauthenticated \
                -o Acquire::AllowInsecureRepositories=true \
                curl ca-certificates git build-essential python3
        elif command -v apk >/dev/null 2>&1; then
            apk add --no-cache curl bash ca-certificates git build-base python3
        fi
    fi
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
    uv python install 3.12
    uv venv /opt/sweagent-venv --python 3.12
    if [ ! -d /opt/sweagent-repo ]; then
        git clone --depth 1 https://github.com/SWE-agent/SWE-agent.git /opt/sweagent-repo
    fi
    # Editable install so the package stays anchored to the source tree
    # at ``/opt/sweagent-repo`` rather than getting copied into
    # site-packages. SWE-agent's ``__init__.py`` asserts the existence
    # of sibling dirs (``config/``, ``tools/``, …) that ship as repo
    # data files OUTSIDE the python package — a normal pip install
    # leaves them behind and the CLI dies at import.
    uv pip install --python /opt/sweagent-venv/bin/python -e /opt/sweagent-repo
fi
/opt/sweagent-venv/bin/sweagent --help >/dev/null
"""


class SweAgentHarness(BaseCliHarness):
    """Run Princeton's SWE-agent inside the sandbox."""

    name = "swe-agent"
    sandbox_backend = "docker"
    max_concurrent = 2  # heavier per-run footprint than the CLI agents
    stdout_log_path = "/tmp/swe-agent.log"
    install_timeout = 1800

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        gateway_url = self._container_url(config.base_url)
        api_key = self.gateway_api_key(config, "OPENAI_API_KEY")
        return {
            "OPENAI_API_KEY": api_key,
            "ANTHROPIC_API_KEY": api_key,
            "OPENAI_API_BASE": gateway_url,
            "OPENAI_BASE_URL": gateway_url,
            "ANTHROPIC_BASE_URL": gateway_url.rstrip("/").removesuffix("/v1") or gateway_url,
        }

    def write_configs(
        self,
        task: Task,
        config: AgentConfig,
        env: dict[str, str],
    ) -> None:
        """Write the task instruction to a file ``sweagent run`` can read.

        ``--problem_statement.path`` insists on a file (not inline text),
        so we drop the instruction at ``/tmp/swe-agent-problem.md``.
        """
        instruction = str(task.instruction).strip()
        # Inline heredoc keeps single-quote-in-instruction safe via a
        # randomized marker — same trick as ``BaseCliHarness._heredoc_write``,
        # but the target path is fully resolved so the helper accepts it.
        self._exec_agent(
            self._heredoc_write("/tmp/swe-agent-problem.md", instruction),
            env=env,
        )

    def build_invocation(
        self,
        instruction: str,
        task: Task,
        config: AgentConfig,
    ) -> str:
        # ``--env.deployment.type=local`` runs tools in the current
        # container instead of spawning a nested docker. ``cost_limit=0``
        # disables litellm's cost guard (the gateway-routed model often
        # isn't in litellm's cost table and the run aborts otherwise).
        _, _, qualified = self.ensure_provider_prefix(config.model)
        # ``--agent.max_steps=30`` caps runaway loops — without it,
        # tool failures (e.g. missing python3) drive the agent through
        # hundreds of retries and rack up real cost. 30 is enough for
        # any plausible single-turn task; tune per benchmark.
        return (
            f"{self._cd_prefix(task)}"
            f"/opt/sweagent-venv/bin/sweagent run "
            f"--agent.model.name={shlex.quote(qualified)} "
            f"--agent.model.per_instance_cost_limit=0 "
            f"--agent.model.total_cost_limit=0 "
            f"--agent.max_steps=30 "
            f"--problem_statement.path=/tmp/swe-agent-problem.md "
            f"--env.deployment.type=local "
            f"--output_dir=/tmp/swe-agent-output "
            f"</dev/null 2>&1 | tee {shlex.quote(self.stdout_log_path)}"
        )

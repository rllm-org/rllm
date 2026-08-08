"""Official mini-swe-agent profile for SWE-bench Pro reproduction.

The generic :mod:`rllm.harnesses.mini_swe_agent` harness intentionally tracks
the newest PyPI release and its default interactive configuration. Official
SWE-bench Pro results instead use the mini-swe-agent revision selected by the
benchmark repository, its SWE-bench prompt, 250 model calls, and no cost cap.
"""

from __future__ import annotations

import re
import shlex

from rllm.data.swebench_pro_builder import OFFICIAL_MINI_SWE_AGENT_REVISION
from rllm.harnesses.mini_swe_agent import MiniSweAgentHarness
from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, Task

_VENV_DIR = "/opt/rllm/swebench-pro-mini"
_REVISION_FILE = f"{_VENV_DIR}/.official-revision"
_ARCHIVE_URL = f"https://github.com/scaleapi/mini-swe-agent/archive/{OFFICIAL_MINI_SWE_AGENT_REVISION}.tar.gz"

_INSTALL_SCRIPT = rf"""
set -e
export DEBIAN_FRONTEND=noninteractive

if ! command -v curl >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -y -qq curl ca-certificates
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache curl bash ca-certificates
    fi
fi

if ! command -v uv >/dev/null 2>&1; then
    for attempt in 1 2 3 4 5; do
        if curl -LsSf https://astral.sh/uv/install.sh | sh; then
            break
        fi
        if [ "$attempt" -eq 5 ]; then
            echo "uv install failed after 5 attempts" >&2
            exit 1
        fi
        sleep $((attempt * 3))
    done
fi
export PATH="$HOME/.local/bin:$PATH"

installed_revision="$(cat {_REVISION_FILE} 2>/dev/null || true)"
if [ "$installed_revision" != "{OFFICIAL_MINI_SWE_AGENT_REVISION}" ]; then
    uv venv --python 3.12 --clear {_VENV_DIR}
    uv pip install --python {_VENV_DIR}/bin/python "{_ARCHIVE_URL}"
    printf '%s\n' "{OFFICIAL_MINI_SWE_AGENT_REVISION}" > {_REVISION_FILE}
fi
"""


class SwebenchProMiniHarness(MiniSweAgentHarness):
    """Run the mini-swe-agent profile used by the official Pro release."""

    name = "swebench-pro-mini"
    max_steps = 250
    temperature = 0.0
    cost_limit = 0.0
    stdout_log_path = "/tmp/swebench-pro-mini.log"

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def write_configs(self, sandbox: Sandbox, task: Task, config: AgentConfig, env: dict[str, str]) -> None:
        super().write_configs(sandbox, task, config, env)
        task_config = task.metadata.get("metadata", {}) or {}
        revision = str(task.metadata.get("official_mini_swe_agent_revision") or task_config.get("official_mini_swe_agent_revision") or OFFICIAL_MINI_SWE_AGENT_REVISION)
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Invalid official mini-swe-agent revision: {revision!r}")

        archive_url = f"https://github.com/scaleapi/mini-swe-agent/archive/{revision}.tar.gz"
        ensure_revision = (
            'export PATH="$HOME/.local/bin:$PATH"; '
            f'installed_revision="$(cat {_REVISION_FILE} 2>/dev/null || true)"; '
            f'if [ "$installed_revision" != "{revision}" ]; then '
            f"uv venv --python 3.12 --clear {_VENV_DIR}; "
            f"uv pip install --python {_VENV_DIR}/bin/python {shlex.quote(archive_url)}; "
            f"printf '%s\\n' {shlex.quote(revision)} > {_REVISION_FILE}; "
            "fi"
        )
        self._exec_agent(sandbox, ensure_revision, timeout=self.install_timeout, env=env)

    def build_invocation(self, instruction: str, task: Task, config: AgentConfig) -> str:
        _, _, qualified = self.ensure_provider_prefix(config.model)
        workdir = str(task.metadata.get("workdir") or "/app")
        workdir_q = shlex.quote(workdir)

        # Scale's official prompt and environment config refer to /testbed,
        # while SWE-bench Pro images keep the checkout at /app. Make those two
        # paths identical without modifying the upstream prompt/config.
        prepare_testbed = f'if [ -e /testbed ] && [ "$(readlink -f /testbed)" != "$(readlink -f {workdir_q})" ]; then rm -rf /testbed; fi; if [ ! -e /testbed ]; then ln -s {workdir_q} /testbed; fi; '
        return (
            f"{self._cd_prefix(task)}"
            f"{prepare_testbed}"
            f"{_VENV_DIR}/bin/mini-swe-agent --yolo "
            f"--config=swebench "
            f"--model={shlex.quote(qualified)} "
            f"--task={shlex.quote(instruction)} "
            f"--cost-limit=0 "
            f"--exit-immediately "
            f"2>&1 | tee {shlex.quote(self.stdout_log_path)}"
        )

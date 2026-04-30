"""MiniSweAgentHarness: runs the mini-swe-agent CLI inside the sandbox.

mini-swe-agent uses litellm under the hood, so it picks up
``OPENAI_API_BASE`` for OpenAI-shaped backends and a model-derived
provider key (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` / …). The
gateway routes by model name regardless.
"""

from __future__ import annotations

import json
import logging
import os
import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.types import AgentConfig, Step, Task

logger = logging.getLogger(__name__)

_TRAJECTORY_PATH = "/tmp/mini-swe-agent.trajectory.json"

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
if ! command -v mini-swe-agent >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -y -qq curl ca-certificates git python3 python3-venv
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache curl bash ca-certificates git python3 py3-pip
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

        # Forward the provider key litellm will look up. The gateway
        # re-stamps auth before forwarding to the real provider, so
        # placeholders work too.
        api_var = _provider_key_var(config.model)
        env[api_var] = os.environ.get(api_var, "sk-rllm-gateway")
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
        api_key = env.get(api_var, "sk-rllm-gateway")

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
        # Run as agent — file lives under the user's HOME, not /root, when
        # the harness's ``agent_user`` is set.
        path = "$HOME/.config/mini-swe-agent/.env"
        # Use a literal $HOME expansion via the shell, since heredoc_write
        # quotes its target — write to a known absolute location instead.
        self._exec_agent(
            f"mkdir -p $HOME/.config/mini-swe-agent && cat > {path} << 'MSWEA_DOTENV_EOF'\n{content}\nMSWEA_DOTENV_EOF",
            env=env,
        )

    def parse_episode(self, stdout: str, task: Task, config: AgentConfig) -> list[Step]:
        """Parse mini-swe-agent's per-turn ``messages`` array into rLLM Steps.

        mini-swe-agent writes its native trajectory to
        ``/tmp/mini-swe-agent.trajectory.json`` (set via ``--output``).
        Each ``messages[i]`` is one role-tagged turn; we collapse every
        assistant→tool pair into a single :class:`Step` whose ``input``
        is the assistant content (LLM response) and ``output`` is the
        tool's stdout. The first user message becomes the seed Step's
        input.

        On any read/parse failure we fall back to one Step with the raw
        stdout so the trial isn't lost — same as the base default.
        """
        if self.sandbox is None:
            return super().parse_episode(stdout, task, config)
        try:
            raw = self.sandbox.exec(
                f"cat {shlex.quote(_TRAJECTORY_PATH)}",
                user=self.agent_user,
            )
            traj = json.loads(raw)
        except Exception as e:
            logger.warning("mini-swe-agent: failed to read %s: %s", _TRAJECTORY_PATH, e)
            return super().parse_episode(stdout, task, config)

        messages = traj.get("messages") or []
        steps: list[Step] = []
        pending_user: str | None = None
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            # Normalize content (may be string, list of parts, or None).
            if isinstance(content, list):
                content = "\n".join(p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content)
            elif content is None:
                content = ""
            else:
                content = str(content)

            if role == "user":
                pending_user = content
                continue
            if role == "assistant":
                steps.append(Step(input=pending_user or "", output=content))
                pending_user = None
                continue
            if role == "tool" and steps:
                # Stitch the tool result onto the previous assistant step
                # as metadata so the gateway-trace + tool-output story lives
                # together in one Step.
                last = steps[-1]
                last.metadata = {**(last.metadata or {}), "tool_output": content}

        return steps or super().parse_episode(stdout, task, config)

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
            f"--output={shlex.quote(_TRAJECTORY_PATH)} "
            f"--exit-immediately "
            f"2>&1 | tee {shlex.quote(self.stdout_log_path)}"
        )

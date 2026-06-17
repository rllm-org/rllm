"""ZeroClawHarness: runs the ZeroClaw CLI agent inside the sandbox.

ZeroClaw (https://github.com/zeroclaw-labs/zeroclaw) is a single Rust binary
"autonomous AI personal assistant". Unlike env-var-driven CLIs (mini-swe-agent)
it is **config-file driven**: it reads ``~/.zeroclaw/config.toml`` for provider
credentials, model, and autonomy policy. So the load-bearing method here is
:meth:`write_configs`, which emits that TOML pointing ZeroClaw's OpenAI-
compatible client at the rLLM gateway.

Three facts established from the ZeroClaw docs/installer:

1. **Install** via the official installer with ``--prebuilt --skip-onboard``
   (downloads a checksum-verified release binary; no Rust toolchain needed).
   The binary lands in ``$CARGO_HOME/bin/zeroclaw`` (``/root/.cargo/bin`` when
   installed as root, which the install hook does).
2. **One-shot, non-interactive** execution is ``zeroclaw agent -m "<prompt>"``
   ("if -m is provided, runs a single turn and exits").
3. **Autonomy** must be ``level = "full"`` for the agent to run shell/filesystem
   tools without interactive approval (``Supervised`` — the default — blocks
   them and the agent stalls in a non-interactive sandbox; cf. issue #851).

``run()`` returns ``None``; the gateway captures every LLM call and the engine
reconstructs the trajectory. ZeroClaw's stdout is tee'd to ``stdout_log_path``
for debugging only.
"""

from __future__ import annotations

import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, Task

# Idempotent install. Bootstraps curl/ca-certs/tar/coreutils (the installer
# needs curl + sha256sum + tar), then runs the official prebuilt installer.
# ``--prebuilt`` avoids a source build; ``--skip-onboard`` skips the interactive
# onboarding (we write config.toml ourselves in write_configs).
_INSTALL_SCRIPT = r"""
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
if ! command -v zeroclaw >/dev/null 2>&1; then
    if ! command -v curl >/dev/null 2>&1 || ! command -v sha256sum >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq 2>/dev/null || true
            apt-get install -y -qq --no-install-recommends curl ca-certificates tar coreutils 2>/dev/null || true
        elif command -v apk >/dev/null 2>&1; then
            apk add --no-cache curl bash ca-certificates tar coreutils
        elif command -v yum >/dev/null 2>&1; then
            yum install -y -q curl ca-certificates tar coreutils
        fi
    fi
    command -v curl >/dev/null 2>&1 \
        || { echo "zeroclaw install: failed to bootstrap curl in sandbox" >&2; exit 1; }
    curl -fsSL https://raw.githubusercontent.com/zeroclaw-labs/zeroclaw/master/install.sh \
        | bash -s -- --prebuilt --skip-onboard
fi
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
grep -q '.cargo/bin' "$HOME/.bashrc" 2>/dev/null \
    || echo 'export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
zeroclaw --version >/dev/null
"""

_CONFIG_PATH = "$HOME/.zeroclaw/config.toml"


def _toml_str(value: str) -> str:
    """Quote a value as a TOML basic string."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class ZeroClawHarness(BaseCliHarness):
    """Run the ZeroClaw CLI agent inside the sandbox."""

    name = "zeroclaw"
    sandbox_backend = "docker"  # overridden to "daytona" via --sandbox-backend
    stdout_log_path = "/tmp/zeroclaw.log"

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        # ZeroClaw resolves credentials in the order: explicit config.toml →
        # env vars (OPENAI_API_KEY/OPENAI_BASE_URL) → fallbacks. We set both:
        # config.toml is authoritative, these env vars are belt-and-suspenders.
        gateway_url = config.base_url
        api_key = self.gateway_api_key(config, "OPENAI_API_KEY")
        return {
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": gateway_url,
            "OPENAI_API_BASE": gateway_url,
        }

    def _config_toml(self, config: AgentConfig) -> str:
        """Render ~/.zeroclaw/config.toml.

        Points ZeroClaw's OpenAI-compatible client at the rLLM gateway session
        URL and grants full autonomy so tools run without interactive approval.

        NOTE: ``base_url`` is the endpoint-override key per ZeroClaw's
        ``ModelProviderConfig`` struct. If a ZeroClaw version rejects it,
        switch to ``uri`` (older docs used that name).
        """
        gateway_url = config.base_url
        api_key = self.gateway_api_key(config, "OPENAI_API_KEY")
        model = config.model
        return "\n".join(
            [
                f"default_provider = {_toml_str('openai')}",
                f"default_model = {_toml_str(model)}",
                "default_temperature = 0.0",
                f"api_key = {_toml_str(api_key)}",
                "",
                "[providers.models.openai]",
                f"api_key = {_toml_str(api_key)}",
                f"base_url = {_toml_str(gateway_url)}",
                f"model = {_toml_str(model)}",
                "",
                "# Full autonomy: run shell/filesystem tools without interactive",
                "# approval (Supervised, the default, stalls in a non-interactive",
                "# sandbox — cf. zeroclaw issue #851).",
                "[autonomy]",
                f"level = {_toml_str('full')}",
                "workspace_only = false",
                "",
                "[gateway]",
                "require_pairing = false",
                "allow_public_bind = false",
                "",
                "[heartbeat]",
                "enabled = false",
                "",
                "[tunnel]",
                'provider = "none"',
                "",
            ]
        )

    def write_configs(self, sandbox: Sandbox, task: Task, config: AgentConfig, env: dict[str, str]) -> None:
        content = self._config_toml(config)
        # Write inline (not via _heredoc_write) so $HOME expands — the helper
        # single-quotes the path which would kill the expansion.
        self._exec_agent(
            sandbox,
            f"mkdir -p $HOME/.zeroclaw && cat > {_CONFIG_PATH} << 'ZC_CONFIG_EOF'\n{content}\nZC_CONFIG_EOF",
            env=env,
        )
        # ZeroClaw operates in its OWN workspace ($HOME/.zeroclaw/workspace),
        # not the shell cwd. rLLM uploads the task fixtures to the task workdir
        # (e.g. /workspace), so without this the agent can't see them. Point
        # ZeroClaw's workspace at the task workdir via a symlink so its
        # file_read/shell tools operate on the fixtures.
        #
        # (We deliberately use a symlink rather than the ZEROCLAW_WORKSPACE env
        # var: that var has version-dependent, buggy config-location coupling —
        # upstream issue #5465 / PR #731 — and setting it made ZeroClaw exit at
        # startup before any LLM call in testing. The symlink is robust.)
        workdir = task.metadata.get("workdir")
        if workdir:
            self._exec_agent(
                sandbox,
                f'rm -rf "$HOME/.zeroclaw/workspace" && ln -sfn {shlex.quote(workdir)} "$HOME/.zeroclaw/workspace"',
                env=env,
            )

    # Prepended to the task instruction so the agent knows its data lives in
    # the workspace. The symlink (write_configs) makes the fixtures *reachable*,
    # but ZeroClaw still doesn't *look*: tested without this preamble, the agent
    # replies "I don't have access to your email account" and makes a single LLM
    # call (reward 0) instead of exploring. With it, it `ls`/`file_read`s the
    # fixtures and completes (4+ tool steps, reward 1). This mirrors how SWE
    # harnesses must tell the agent about the repo — it shapes *where to look*,
    # not *what answer to give*, so it keeps failures attributable to reasoning.
    _WORKSPACE_PREAMBLE = (
        "You are an autonomous agent working in a sandbox. All files and data needed for this "
        "task are already present in your workspace (your working directory). Begin by exploring "
        "them with your shell and file_read tools (e.g. `ls -R`), then act on them to complete the "
        "task. Do not claim you lack access — the data is in the workspace.\n\nTask:\n"
    )

    def build_invocation(self, instruction: str, task: Task, config: AgentConfig) -> str:
        # ``-m`` runs a single non-interactive turn and exits. ``</dev/null``
        # guarantees no stdin block if the binary probes for a TTY. The agent's
        # workspace (symlinked to the task workdir in write_configs) holds the
        # fixtures; the preamble tells it to read them with file_read/shell.
        prompt = self._WORKSPACE_PREAMBLE + instruction
        return f'{self._cd_prefix(task)}export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"; zeroclaw agent -m {shlex.quote(prompt)} </dev/null 2>&1 | tee {shlex.quote(self.stdout_log_path)}'

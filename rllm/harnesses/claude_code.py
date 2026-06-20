"""ClaudeCodeHarness: runs the Claude Code CLI inside the sandbox.

Subclass of :class:`BaseCliHarness`, so it inherits the install/run
lifecycle, gateway-trace-driven trajectory wiring (``run() -> None``),
host-loopback rewrite, and bearer-token auth handling.

The install + invocation shape mirrors harbor's verified-working
``claude-code`` agent (``harbor/src/harbor/agents/installed/claude_code.py``).
Two facts are load-bearing:

1. **Use the official Anthropic installer**
   (``curl https://claude.ai/install.sh | bash``) on apt/yum distros;
   fall back to ``npm i -g @anthropic-ai/claude-code`` only on Alpine
   (where the install script's binary doesn't run). The installer
   drops ``claude`` into ``$HOME/.local/bin`` — the agent's PATH must
   pick that up at run time.
2. **``IS_SANDBOX=1`` is required for ``--permission-mode=bypassPermissions``**.
   Without it the CLI ignores the flag and silently exits without
   making any LLM calls (which is what produced the previous "ran but
   no traces" symptom).
"""

from __future__ import annotations

import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.types import AgentConfig, Task

# Install strategy mirrors harbor: Alpine → npm (the install script's
# musl-linked binary fails on Alpine), everyone else → official curl
# script (more reliable than nvm bootstrap, doesn't need a node toolchain).
#
# The apt branch is hardened against two recurring container quirks:
#  1. arm64 ``ports.ubuntu.com`` ``InRelease`` GPG flakes (Ubuntu noble
#     on Docker Desktop / macOS): ``apt-get update`` exits non-zero with
#     ``is not signed``, ``&&`` short-circuits, curl never installs.
#     Fix: ``-o Acquire::AllowInsecureRepositories=true`` plus
#     ``--allow-unauthenticated``.
#  2. Tight overlay space on tasks where ``/var`` is small (the
#     hello-world image with ``storage_mb = 10240`` still hits
#     ``E: You don't have enough free space in /var/cache/apt/archives/``
#     during install). Fix: redirect apt's download cache + lists to
#     ``/tmp`` (tmpfs, plenty of room), so the real root only stores
#     the extracted binaries.
_INSTALL_SCRIPT = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# Redirect apt's bulky temp dirs to /tmp so /var doesn't fill up.
_APT_OPTS=(
    -o Dir::Cache::archives=/tmp/rllm-apt-archives
    -o Dir::State::lists=/tmp/rllm-apt-lists
    -o Acquire::AllowInsecureRepositories=true
    -o Acquire::AllowDowngradeToInsecureRepositories=true
    -o Acquire::Check-Valid-Until=false
)

if ! { export PATH="$HOME/.local/bin:$PATH"; command -v claude >/dev/null 2>&1; }; then
    if ! command -v curl >/dev/null 2>&1; then
        if command -v apk >/dev/null 2>&1; then
            apk add --no-cache curl bash nodejs npm ca-certificates
        elif command -v apt-get >/dev/null 2>&1; then
            mkdir -p /tmp/rllm-apt-archives/partial /tmp/rllm-apt-lists/partial
            apt-get "${_APT_OPTS[@]}" update -qq 2>/dev/null || true
            apt-get "${_APT_OPTS[@]}" install -y -qq \
                --no-install-recommends --allow-unauthenticated \
                curl ca-certificates
            rm -rf /tmp/rllm-apt-archives /tmp/rllm-apt-lists
        elif command -v yum >/dev/null 2>&1; then
            yum install -y -q curl ca-certificates
        fi
    fi
    command -v curl >/dev/null 2>&1 \
        || { echo "claude-code install: failed to bootstrap curl in sandbox" >&2; exit 1; }
    if command -v apk >/dev/null 2>&1; then
        if ! command -v npm >/dev/null 2>&1; then
            apk add --no-cache nodejs npm
        fi
        npm install -g @anthropic-ai/claude-code
    else
        # Anthropic's official installer lays ``claude`` into ``$HOME/.local/bin``.
        curl -fsSL https://claude.ai/install.sh | bash
    fi
fi
# Persist PATH for future agent execs; current root-exec also needs it
# to satisfy the ``claude --version`` smoke check below.
grep -q 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null \
    || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
export PATH="$HOME/.local/bin:$PATH"
claude --version >/dev/null
"""

# Per-task CLAUDE_CONFIG_DIR keeps Claude Code's local state (debug logs,
# project sessions, statsig) out of $HOME/.claude — useful when many runs
# share an image, and mandatory for read-only $HOME setups.
_CLAUDE_CONFIG_DIR = "/tmp/claude-config"


class ClaudeCodeHarness(BaseCliHarness):
    """Run Anthropic's Claude Code CLI inside the sandbox."""

    name = "claude-code"
    sandbox_backend = "docker"
    stdout_log_path = "/tmp/claude-code.log"

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        # Anthropic's SDK appends ``/v1/messages`` itself, so strip a
        # trailing ``/v1`` from the gateway URL or it doubles up.
        gateway_url = config.base_url
        anthropic_url = gateway_url.rstrip("/").removesuffix("/v1") or gateway_url
        api_key = self.gateway_api_key(config, "ANTHROPIC_API_KEY")
        model = config.model

        env: dict[str, str] = {
            "ANTHROPIC_BASE_URL": anthropic_url,
            "ANTHROPIC_API_KEY": api_key,
            "ANTHROPIC_MODEL": model,
            # IS_SANDBOX=1 is the gate for ``--permission-mode=bypassPermissions``
            # to actually take effect. Without it the CLI exits silently
            # on the first tool call that would touch the filesystem.
            "IS_SANDBOX": "1",
            # Isolate Claude Code state to a per-task directory so the
            # harness doesn't pollute $HOME between runs.
            "CLAUDE_CONFIG_DIR": _CLAUDE_CONFIG_DIR,
            # Skip the telemetry POSTs the CLI fires on startup — they
            # add latency and have nothing to do with the eval.
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            # Allow long-running tool calls (file watches, build steps)
            # to keep going across agent turns instead of being killed
            # on the first idle window.
            "FORCE_AUTO_BACKGROUND_TASKS": "1",
            "ENABLE_BACKGROUND_TASKS": "1",
            # When the gateway redirects to a non-Anthropic model, the
            # CLI's internal "auto" model selection (sonnet/opus/haiku
            # aliases used for sub-agents and resumed sessions) still
            # routes — point all three at the chosen model.
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
        }
        return env

    def build_invocation(
        self,
        instruction: str,
        task: Task,
        config: AgentConfig,
    ) -> str:
        # Match harbor's verified flag shape:
        #   claude --verbose --output-format=stream-json
        #          --permission-mode=bypassPermissions
        #          --print -- <prompt>
        # ``--print`` is the non-interactive flag (printable result on
        # stdout, then exit). ``--`` is the flag terminator so prompts
        # beginning with ``-`` aren't reparsed as options.
        # ``mkdir -p $CLAUDE_CONFIG_DIR`` matches harbor's pre-run setup —
        # the CLI ENOENTs trying to write its debug log if the dir is
        # missing.
        return (
            f"{self._cd_prefix(task)}"
            f'export PATH="$HOME/.local/bin:$PATH"; '
            f"mkdir -p {shlex.quote(_CLAUDE_CONFIG_DIR)}; "
            f"claude --verbose --output-format=stream-json "
            f"--permission-mode=bypassPermissions "
            f"--print -- {shlex.quote(instruction)} "
            f"</dev/null 2>&1 | tee {shlex.quote(self.stdout_log_path)}"
        )

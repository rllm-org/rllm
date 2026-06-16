"""CodexHarness: runs the OpenAI Codex CLI inside the sandbox.

This implementation mirrors the proven pattern from
``harbor/src/harbor/agents/installed/codex.py`` (verified working against
Codex CLI 0.118.0+). Two non-obvious facts drive the shape:

1. **Codex reads auth from a JSON file, not from ``OPENAI_API_KEY``
   alone.** Recent Codex versions resolve credentials from
   ``$CODEX_HOME/auth.json`` (schema: ``{"OPENAI_API_KEY": "..."}``).
   Setting only the env var leaves the CLI looking unauthenticated.
2. **Codex 0.118+ ignores ``OPENAI_BASE_URL`` and instead reads
   ``openai_base_url`` from ``$CODEX_HOME/config.toml``.** Routing
   through the rLLM gateway therefore requires writing that key — env
   var alone goes straight to api.openai.com.

We set ``CODEX_HOME`` to ``/tmp/codex-home`` so auth/config never touch
``$HOME/.codex`` (cleaner in shared/sandbox environments).

``run()`` returns ``None``; the gateway captures every LLM call and
the engine builds the trajectory.
"""

from __future__ import annotations

import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, Task

# apt branch hardened against arm64 ubuntu-ports GPG flakes — same
# rationale as claude_code.py's install script. Without it, ``apt-get
# update`` exits non-zero, ``&&`` short-circuits, curl never installs,
# and the script dies with ``curl: command not found``.
_INSTALL_SCRIPT = r"""
set -e
export DEBIAN_FRONTEND=noninteractive
if ! command -v codex >/dev/null 2>&1; then
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
        || { echo "codex install: failed to bootstrap curl in sandbox" >&2; exit 1; }
    if ! command -v node >/dev/null 2>&1; then
        export NVM_DIR="$HOME/.nvm"
        curl -fsSL -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
        \. "$NVM_DIR/nvm.sh"
        nvm install 22
    fi
    [ -s "$HOME/.nvm/nvm.sh" ] && \. "$HOME/.nvm/nvm.sh"
    npm install -g @openai/codex
fi
codex --version >/dev/null
"""

_CODEX_HOME = "/tmp/codex-home"


class CodexHarness(BaseCliHarness):
    """Run OpenAI's Codex CLI inside the sandbox."""

    name = "codex"
    sandbox_backend = "docker"
    stdout_log_path = "/tmp/codex.log"

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        gateway_url = config.base_url
        return {
            # Custom CODEX_HOME so auth/config files don't touch $HOME/.codex.
            "CODEX_HOME": _CODEX_HOME,
            # OPENAI_API_KEY is still required for some code paths even when
            # auth.json is present — keep it in sync with the auth file.
            "OPENAI_API_KEY": self.gateway_api_key(config, "OPENAI_API_KEY"),
            # Codex 0.118+ ignores this env var (reads openai_base_url from
            # config.toml instead) but earlier versions honored it — set both
            # for forward/backward compat.
            "OPENAI_BASE_URL": gateway_url,
        }

    def write_configs(
        self,
        sandbox: Sandbox,
        task: Task,
        config: AgentConfig,
        env: dict[str, str],
    ) -> None:
        """Write ``$CODEX_HOME/auth.json`` and ``$CODEX_HOME/config.toml``.

        - ``auth.json`` schema: ``{"OPENAI_API_KEY": "..."}`` — Codex reads
          credentials from here. Setting ``OPENAI_API_KEY`` in env alone is
          not enough for ``codex exec`` to pick them up.
        - ``config.toml`` only needs ``openai_base_url = "..."`` to redirect
          the bundled openai provider at our gateway. The custom
          ``model_provider`` / ``model_providers.<id>`` schema is NOT
          required (and was the source of the silent no-op in earlier
          versions of this harness).
        """
        gateway_url = config.base_url
        api_key = env.get("OPENAI_API_KEY", self.gateway_api_key(config, "OPENAI_API_KEY"))

        # JSON has to be escaped enough to survive a single-quoted heredoc
        # marker. The key has no quotes/backslashes in practice (it's a
        # bearer token or sk-... string), but use json.dumps to be safe.
        import json as _json

        auth_json = _json.dumps({"OPENAI_API_KEY": api_key})
        config_toml = f'openai_base_url = "{gateway_url}"\n'

        # Heredoc target paths must leave ``$CODEX_HOME`` UNQUOTED so the
        # shell expands it (same lesson as opencode.py / mini_swe_agent.py
        # — ``_heredoc_write`` single-quotes the path which kills the
        # ``$VAR`` expansion).
        cmd = (
            'mkdir -p "$CODEX_HOME" && '
            "cat > \"$CODEX_HOME/auth.json\" << '_RLLM_CODEX_AUTH_EOF'\n"
            f"{auth_json}\n"
            "_RLLM_CODEX_AUTH_EOF\n"
            "cat > \"$CODEX_HOME/config.toml\" << '_RLLM_CODEX_CFG_EOF'\n"
            f"{config_toml}"
            "_RLLM_CODEX_CFG_EOF"
        )
        self._exec_agent(sandbox, cmd, env=env)

    def build_invocation(
        self,
        instruction: str,
        task: Task,
        config: AgentConfig,
    ) -> str:
        # Match harbor's verified invocation exactly:
        #   codex exec
        #     --dangerously-bypass-approvals-and-sandbox  # skip Codex's inner sandbox
        #     --skip-git-repo-check                       # task workdirs aren't git repos
        #     --model <bare>                              # bare id (no provider/)
        #     --json --enable unified_exec                # required for non-interactive
        #     -- <prompt>                                 # positional, after flag terminator
        # The ``--`` is load-bearing: without it Codex tries to parse the
        # instruction as flags when it starts with ``-``.
        _, model_id, _ = self.ensure_provider_prefix(config.model)
        return (
            f"{self._cd_prefix(task)}"
            f". $HOME/.nvm/nvm.sh 2>/dev/null; "
            f"codex exec "
            f"--dangerously-bypass-approvals-and-sandbox "
            f"--skip-git-repo-check "
            f"--model {shlex.quote(model_id)} "
            f"--json "
            f"--enable unified_exec "
            f"-- {shlex.quote(instruction)} "
            f"</dev/null 2>&1 | tee {shlex.quote(self.stdout_log_path)}"
        )

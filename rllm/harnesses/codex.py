"""CodexHarness: runs the OpenAI Codex CLI inside the sandbox.

Current Codex CLI is Responses-API-only: ``wire_api = "chat"`` is
hard-rejected at startup with ``no longer supported`` (see
https://github.com/openai/codex/discussions/7782). Codex POSTs to
``/v1/responses``, NOT ``/v1/chat/completions``.

The gateway's catch-all proxy still records a TraceRecord per call
(role, finish_reason, token counts) but the chat-completion parser
can't extract the message *content* from Responses API output
(``output[].content[].text`` shape vs ``choices[0].message.content``).
Net effect: the run succeeds, reward is correct, ``rllm view`` shows
the right number of steps — but message content reads as ``None``
until the gateway grows a Responses-API parser (separate piece of
work; see also the harness docs).

Three other non-obvious facts drive the shape:

1. **Codex reads auth from a JSON file**, ``$CODEX_HOME/auth.json``
   (schema: ``{"OPENAI_API_KEY": "..."}``). Setting only the env var
   leaves the CLI looking unauthenticated.
2. **The bundled ``openai`` provider is locked** in ways that block
   the rllm gateway — registering an explicit custom provider via
   ``model_provider`` + ``[model_providers.<id>]`` is the reliable
   routing path.
3. **Codex ignores ``OPENAI_BASE_URL`` env**. The custom-provider
   block's ``base_url`` is the only routing knob that takes effect.

``CODEX_HOME`` is set to ``/tmp/codex-home`` so auth/config never
touch ``$HOME/.codex``. ``run()`` returns ``None`` — gateway-captured
traces drive Episode enrichment.
"""

from __future__ import annotations

import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
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
_RLLM_PROVIDER_ID = "rllm-gateway"


class CodexHarness(BaseCliHarness):
    """Run OpenAI's Codex CLI inside the sandbox."""

    name = "codex"
    sandbox_backend = "docker"
    max_concurrent = 4
    stdout_log_path = "/tmp/codex.log"

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        gateway_url = self._container_url(config.base_url)
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
        task: Task,
        config: AgentConfig,
        env: dict[str, str],
    ) -> None:
        """Write ``$CODEX_HOME/auth.json`` and ``$CODEX_HOME/config.toml``.

        - ``auth.json`` schema: ``{"OPENAI_API_KEY": "..."}`` — Codex
          reads credentials from here.
        - ``config.toml`` declares the rLLM gateway as a custom provider
          with ``wire_api = "chat"`` and selects it via ``model_provider``.
          Critical: overriding the bundled ``openai`` provider in place
          doesn't change its locked-in ``responses`` wire api in current
          Codex versions, so a separate provider id is required.
        """
        gateway_url = self._container_url(config.base_url)
        api_key = env.get("OPENAI_API_KEY", self.gateway_api_key(config, "OPENAI_API_KEY"))
        _, model_id, _ = self.ensure_provider_prefix(config.model)

        import json as _json

        auth_json = _json.dumps({"OPENAI_API_KEY": api_key})

        # Hand-rolled TOML — schema is tiny and the stdlib has no
        # writer. ``wire_api = "responses"`` is the only value current
        # Codex CLI versions accept; ``"chat"`` is hard-rejected at
        # startup. This means the gateway sees /v1/responses traffic
        # which it currently doesn't parse into TraceRecords (file-
        # level docstring covers the implications).
        config_toml = (
            f'model = "{model_id}"\n'
            f'model_provider = "{_RLLM_PROVIDER_ID}"\n'
            f"\n"
            f"[model_providers.{_RLLM_PROVIDER_ID}]\n"
            f'name = "rLLM Gateway"\n'
            f'base_url = "{gateway_url}"\n'
            f'env_key = "OPENAI_API_KEY"\n'
            f'wire_api = "responses"\n'
        )

        # Heredoc target paths must leave ``$CODEX_HOME`` UNQUOTED so
        # the shell expands it at exec time.
        cmd = (
            'mkdir -p "$CODEX_HOME" && '
            "cat > \"$CODEX_HOME/auth.json\" << '_RLLM_CODEX_AUTH_EOF'\n"
            f"{auth_json}\n"
            "_RLLM_CODEX_AUTH_EOF\n"
            "cat > \"$CODEX_HOME/config.toml\" << '_RLLM_CODEX_CFG_EOF'\n"
            f"{config_toml}"
            "_RLLM_CODEX_CFG_EOF"
        )
        self._exec_agent(cmd, env=env)

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

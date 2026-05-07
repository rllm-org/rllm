"""Base class for harnesses that wrap a CLI tool installed inside a sandbox.

The pattern is the same across opencode, mini-swe-agent, claude-code, codex:

1. ``install`` — once per sandbox, package-manager + npm/uv/curl install of
   the CLI binary. Idempotent.
2. ``build_env`` — assemble the env dict (gateway base URL, auth, model
   name) that the CLI process will read.
3. ``write_configs`` — some CLIs also need a config file inside the
   sandbox (opencode requires ``opencode.json``; mini-swe-agent requires
   a ``~/.config/mini-swe-agent/.env`` dotenv). Hook with default no-op.
4. ``build_invocation`` — return the shell command to run the CLI.

``run()`` execs that command and returns ``None``. The gateway sits in
front of every LLM call the CLI makes, so wire-level traces flow into
the engine and ``_coerce_to_episode`` (in :mod:`rllm.types`) builds the
Episode with one empty Trajectory whose Steps get populated from those
traces during enrichment. No manual stdout parsing or Step construction
is needed here — the CLI's stdout is captured for debugging via the
``stdout_log_path`` tee but is otherwise ignored.

The gateway URL the harness sees is already stamped with
``/sessions/<session_uid>/v1`` upstream, so every LLM call is tagged
with this task's session in the gateway sqlite. The harness just
propagates ``config.base_url`` through the appropriate env var
(``ANTHROPIC_BASE_URL`` / ``OPENAI_BASE_URL``).
"""

from __future__ import annotations

import logging
import re
import shlex
import uuid
from abc import abstractmethod

from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow
from rllm.types import AgentConfig, Task

logger = logging.getLogger(__name__)


class BaseCliHarness(SandboxedAgentFlow):
    """Run a CLI agent (opencode, mini-swe-agent, claude-code, …) inside the sandbox.

    Subclasses fill in the install command, env var dict, optional config
    files, and the invocation. ``run`` returns ``None``; the gateway
    captures every LLM call and the engine builds the trajectory.
    """

    # Display name used by the agent registry.
    name: str = "cli"
    # Default sandbox backend — overridable via class attr or kwargs.
    sandbox_backend: str = "docker"
    # Default container image. Tasks usually override via per-task images.
    image: str = "python:3.11-slim"
    # Concurrency hint for the eval runner.
    max_concurrent: int = 4
    # Container user the CLI runs as. ``None`` uses the image default.
    agent_user: str | None = None
    # Path inside the sandbox where the CLI's stdout is teed.
    stdout_log_path: str = "/tmp/agent-stdout.log"
    # Per-call timeouts (seconds). Tasks may override via metadata.
    install_timeout: int = 600
    run_timeout: int = 1800

    # ---------------------------------------------------------------------
    # Sandbox helpers
    # ---------------------------------------------------------------------

    def _exec_root(self, command: str, timeout: float | None = None) -> str:
        """Run *command* as root inside the sandbox."""
        if self.sandbox is None:
            raise RuntimeError(f"{type(self).__name__} requires a sandbox.")
        return self.sandbox.exec(command, timeout=timeout, user="root")

    def _exec_agent(
        self,
        command: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        """Run *command* as the agent user, with *env* exported.

        Uses ``export`` (not the inline ``K=V cmd`` prefix), because
        invocations like ``cd /workspace && claude ...`` are compound
        commands. Bash's inline assignment only applies to the *first*
        command in the chain — ``ANTHROPIC_API_KEY`` would be set for
        ``cd`` and gone by the time the CLI runs, leaving the agent
        looking like it had no auth even though we passed it.
        """
        if self.sandbox is None:
            raise RuntimeError(f"{type(self).__name__} requires a sandbox.")
        if env:
            exports = "; ".join(f"export {k}={shlex.quote(v)}" for k, v in env.items() if v is not None)
            command = f"{exports}; {command}"
        return self.sandbox.exec(command, timeout=timeout, user=self.agent_user)

    @staticmethod
    def infer_provider(model_name: str) -> str:
        """Map a bare model name to its likely provider slug.

        rLLM's setup configures bare model names (``gpt-5.4-mini``,
        ``claude-opus-4-1``) but several CLIs (opencode, mini-swe-agent)
        require ``provider/model`` form.

        Returns the lowercase provider slug. Defaults to ``openai`` for
        unknown patterns — works for ``gpt-*``/``o1``/``o3`` and routes
        cleanly through any OpenAI-compatible proxy.
        """
        name = model_name.lower()
        if any(k in name for k in ("claude", "haiku", "sonnet", "opus")):
            return "anthropic"
        if "gemini" in name or "gemma" in name:
            return "google"
        if "deepseek" in name:
            return "deepseek"
        if "grok" in name:
            return "xai"
        if "mistral" in name or "mixtral" in name:
            return "mistral"
        if "qwen" in name:
            return "openai"  # Qwen via OpenAI-compatible endpoints
        return "openai"

    @staticmethod
    def gateway_api_key(config: AgentConfig, fallback_env_var: str) -> str:
        """Return the API key to inject into the sandbox for *fallback_env_var*.

        When the eval gateway is exposed publicly it generates an
        ``inbound_auth_token`` and stamps it on
        ``config.metadata["gateway_auth_token"]``. Every provider key
        the harness writes into the sandbox env (``OPENAI_API_KEY``,
        ``ANTHROPIC_API_KEY``, …) must be that bearer token, because
        that's what the gateway's middleware checks. The gateway then
        replaces the auth header with the route's pre-resolved upstream
        auth header before forwarding.

        Loopback gateways (no token) keep the current behaviour: pass
        the user's real key through, or a placeholder if unset.
        """
        token = (config.metadata or {}).get("gateway_auth_token")
        if token:
            return token
        import os

        return os.environ.get(fallback_env_var, "sk-rllm-gateway")

    @classmethod
    def ensure_provider_prefix(cls, model_name: str) -> tuple[str, str, str]:
        """Return ``(provider, model_id, qualified_name)`` for *model_name*.

        Accepts both bare names (``gpt-4o``) and pre-qualified names
        (``openai/gpt-4o``); inference fills in the provider when missing.
        """
        if "/" in model_name:
            provider, model_id = model_name.split("/", 1)
        else:
            provider = cls.infer_provider(model_name)
            model_id = model_name
        return provider, model_id, f"{provider}/{model_id}"

    def _container_url(self, url: str) -> str:
        """Rewrite host loopback addresses for in-container reachability.

        The gateway binds to 127.0.0.1 on the host. From inside a Docker
        container, that loopback addresses the container itself, not the
        host. Docker Desktop (macOS/Windows) resolves the magic hostname
        ``host.docker.internal`` to the host; on Linux Docker 20.10+ the
        same hostname works when the container is started with
        ``--add-host=host.docker.internal:host-gateway``.

        For the ``local`` and ``modal`` backends this is a no-op.
        """
        if self.sandbox_backend != "docker":
            return url
        return re.sub(
            r"(https?://)(?:127\.0\.0\.1|localhost)(:\d+|/|$)",
            r"\1host.docker.internal\2",
            url,
        )

    @staticmethod
    def _cd_prefix(task: Task) -> str:
        """Return ``cd <workdir> && `` only when the task explicitly sets ``workdir``.

        Without this, a hardcoded ``cd /workspace`` overrides whatever
        ``WORKDIR`` the task's Dockerfile declared, so verifiers that
        check ``/app/foo`` (the Dockerfile WORKDIR) instead of
        ``/workspace/foo`` fail even when the agent did the right thing.
        Tasks that need a specific workdir set it via
        ``[environment].workdir`` in task.toml → ``task.metadata['workdir']``.
        """
        workdir = task.metadata.get("workdir")
        return f"cd {shlex.quote(workdir)} && " if workdir else ""

    @staticmethod
    def _heredoc_write(remote_path: str, content: str) -> str:
        """Build a shell command that writes *content* to *remote_path*.

        Uses a unique heredoc marker so embedded ``EOF`` strings in the
        content can't terminate it. The parent directory is created.

        ``remote_path`` MUST be an absolute, fully-resolved path — shell
        variables like ``$HOME`` won't expand because the path is
        single-quoted to prevent injection. Callers that need ``$HOME``
        should hand-roll the heredoc with the path *unquoted* in the
        shell command.
        """
        if "$" in remote_path:
            raise ValueError(f"_heredoc_write requires a fully-resolved path; got {remote_path!r}. Single-quoting kills $VAR expansion — write the heredoc inline.")
        marker = f"_RLLM_CLI_EOF_{uuid.uuid4().hex[:8]}"
        parent = shlex.quote(remote_path.rsplit("/", 1)[0] or "/")
        path_q = shlex.quote(remote_path)
        return f"mkdir -p {parent} && cat > {path_q} << '{marker}'\n{content}\n{marker}"

    # ---------------------------------------------------------------------
    # Hooks subclasses implement
    # ---------------------------------------------------------------------

    @abstractmethod
    def install_script(self) -> str:
        """Return a shell script that installs the CLI in the sandbox.

        Runs as root via ``on_sandbox_ready``. Must be idempotent — the
        hook may fire on a container where the CLI is already present.
        """

    @abstractmethod
    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        """Return the env vars the CLI needs (auth, base URL, model)."""

    def write_configs(
        self,
        task: Task,
        config: AgentConfig,
        env: dict[str, str],
    ) -> None:
        """Hook: write any in-sandbox config files the CLI requires.

        Default no-op. Override for opencode (opencode.json) and
        mini-swe-agent (~/.config/mini-swe-agent/.env).
        """

    @abstractmethod
    def build_invocation(
        self,
        instruction: str,
        task: Task,
        config: AgentConfig,
    ) -> str:
        """Return the shell command that runs the CLI on *instruction*.

        The command should tee its stdout to ``self.stdout_log_path`` so
        operators have a captured log to read for debugging — Steps are
        built from gateway traces, not from this stdout.
        """

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    def on_sandbox_ready(self, task: dict, config: AgentConfig) -> None:  # noqa: ARG002
        """Install the CLI inside the sandbox before the first run."""
        if self.sandbox is None:
            return
        try:
            self._exec_root(self.install_script(), timeout=self.install_timeout)
        except Exception as e:
            raise RuntimeError(f"Failed to install {self.name} in sandbox: {e}") from e

    def run(self, task: Task, config: AgentConfig) -> None:
        """Exec the CLI; let the gateway build the trajectory.

        Returns ``None`` so :func:`rllm.types._coerce_to_episode` builds
        an empty single-trajectory Episode. The engine then enriches its
        Steps from the gateway-captured traces during ``execute_tasks``.
        """
        if self.sandbox is None:
            raise RuntimeError(f"{type(self).__name__} requires a sandbox.")

        env = self.build_env(task, config)
        self.write_configs(task, config, env)

        instruction = str(task.instruction).strip()
        timeout = float(task.metadata.get("agent_timeout", self.run_timeout))
        cmd = self.build_invocation(instruction, task, config)

        try:
            self._exec_agent(cmd, timeout=timeout, env=env)
        except Exception as e:
            # Surface as a warning for operator visibility; the engine
            # still gets None and the gateway traces (if any LLM calls
            # made it through before the failure) drive enrichment.
            logger.warning("%s execution failed: %s", type(self).__name__, e)

        return None

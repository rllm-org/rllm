"""Base class for harnesses that wrap a CLI tool installed inside a sandbox.

The pattern is the same across opencode, codex, claude-code, mini-swe-agent:

1. ``install`` — once per sandbox, package-manager + npm/uv/curl install of
   the CLI binary. Idempotent.
2. ``build_env`` — assemble the env dict (gateway base URL, auth, model
   name) that the CLI process will read.
3. ``write_configs`` — some CLIs also need a config file inside the
   sandbox (codex requires ``config.toml``; opencode requires
   ``opencode.json``). Hook with default no-op.
4. ``build_invocation`` — return the shell command to run the CLI.
5. ``parse_episode`` — turn captured stdout into Steps.

The gateway URL the harness sees is already stamped with
``/sessions/<session_uid>/v1`` by :func:`rllm.eval.runner._stamp_session_in_url`,
so every LLM call from the CLI is tagged with this task's session in the
gateway sqlite. The harness just propagates ``config.base_url`` through
the appropriate env var (``ANTHROPIC_BASE_URL`` / ``OPENAI_BASE_URL``).
"""

from __future__ import annotations

import hashlib
import logging
import re
import shlex
import uuid
from abc import abstractmethod

from rllm.sandbox.protocol import Sandbox
from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

logger = logging.getLogger(__name__)


class BaseCliHarness(SandboxedAgentFlow):
    """Run a CLI agent (opencode, codex, claude-code, …) inside the sandbox.

    Subclasses fill in the install command, env var dict, optional config
    files, and the invocation. Trajectory parsing defaults to one Step
    holding the raw stdout — override for per-CLI structured parsing.
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
        """Run *command* as the agent user, with *env* prepended."""
        if self.sandbox is None:
            raise RuntimeError(f"{type(self).__name__} requires a sandbox.")
        if env:
            prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items() if v is not None)
            command = f"{prefix} {command}"
        return self.sandbox.exec(command, timeout=timeout, user=self.agent_user)

    @staticmethod
    def infer_provider(model_name: str) -> str:
        """Map a bare model name to its likely provider slug.

        rLLM's setup configures bare model names (``gpt-5.4-mini``,
        ``claude-opus-4-1``) but several CLIs (opencode, mini-swe-agent)
        require ``provider/model`` form. This helper mirrors the
        inference that ``rllm.integrations.harbor.trial_helper`` uses,
        so the same conventions apply across both runtime paths.

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
        token = config.metadata.get("gateway_auth_token")
        if token:
            return token
        import os as _os

        return _os.environ.get(fallback_env_var, "sk-rllm-gateway")

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
        import re

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

        Runs as root. Must be idempotent — ``on_sandbox_ready`` may fire
        on a container that already has the CLI from a prior task.
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

        Default no-op. Override for opencode (opencode.json) and codex
        (config.toml + auth.json).
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
        :meth:`parse_episode` can read it back.
        """

    def parse_episode(
        self,
        stdout: str,
        task: Task,
        config: AgentConfig,
    ) -> list[Step]:
        """Convert raw CLI stdout to Steps. Default: one step with the full output.

        Override per-CLI for structured parsing (opencode JSON-lines,
        codex session jsonl, etc.). Note that detailed LLM-call traces
        already live in the gateway sqlite — this trajectory exists for
        debugging and verifier coupling, not for re-deriving prompts.
        """
        return [Step(input=str(task.instruction).strip(), output=stdout)]

    # ---------------------------------------------------------------------
    # Image caching — first task pays cold-install cost, derived image
    # is committed and reused by every subsequent task with the same
    # base image + same install script.
    # ---------------------------------------------------------------------

    def _installed_image_tag(self, base_image: str) -> str:
        """Deterministic tag for the post-install image of this harness.

        Differs from the base when the install script changes, so an
        edit to ``install_script()`` invalidates the cache automatically.
        """
        sig = f"{base_image}\n{self.install_script()}".encode()
        digest = hashlib.sha1(sig).hexdigest()[:10]
        # Sanitize base_image into a tag-safe slug (~24 chars, no slashes).
        slug = re.sub(r"[^a-zA-Z0-9_.-]", "-", base_image)[-24:].lower().lstrip(".-")
        return f"rllm-cli-{self.name}-{slug}-{digest}:installed"

    def _docker_image_exists(self, tag: str) -> bool:
        """Check whether *tag* is already present in the local docker image cache."""
        try:
            import docker
        except ImportError:
            return False
        try:
            docker.from_env().images.get(tag)
            return True
        except Exception:
            return False

    def maybe_use_cached_image(self, base_image: str, backend: str) -> str:
        """If a post-install image for this base+harness exists, use it instead of *base_image*."""
        if backend != "docker":
            return base_image
        cached = self._installed_image_tag(base_image)
        if self._docker_image_exists(cached):
            logger.info("%s: reusing cached image %s", self.name, cached)
            return cached
        return base_image

    def pre_setup(self, sandbox: Sandbox, base_image: str, backend: str) -> None:
        """Install the CLI in the sandbox and (on docker) commit the result.

        Runs before any task-specific files touch the container so the
        committed image stays clean and usable across tasks with the
        same base image.
        """
        try:
            sandbox.exec(self.install_script(), timeout=self.install_timeout, user="root")
        except Exception as e:
            raise RuntimeError(f"Failed to install {self.name} in sandbox: {e}") from e

        if backend != "docker":
            return
        cached = self._installed_image_tag(base_image)
        if self._docker_image_exists(cached):
            return  # We're already running on the cached image.
        if not hasattr(sandbox, "commit"):
            return  # Backend doesn't support commit.
        try:
            sandbox.commit(cached)  # type: ignore[attr-defined]
        except Exception as e:
            # Caching is a perf optimization, not a correctness requirement.
            logger.warning("%s: docker commit to %s failed (will reinstall next task): %s", self.name, cached, e)

    # ---------------------------------------------------------------------
    # Lifecycle wired into SandboxedAgentFlow
    # ---------------------------------------------------------------------

    def on_sandbox_ready(self, task: dict, config: AgentConfig) -> None:  # noqa: ARG002
        """Compatibility shim: install runs in :meth:`pre_setup` now.

        Older callers (the legacy ``SandboxedAgentFlow.setup_sandbox``
        path) still invoke this hook. Re-running ``install_script`` here
        is a fast no-op when the CLI is already on PATH because the
        scripts gate on ``command -v <cli>``.
        """
        if self.sandbox is None:
            return
        try:
            self._exec_root(self.install_script(), timeout=self.install_timeout)
        except Exception as e:
            raise RuntimeError(f"Failed to install {self.name} in sandbox: {e}") from e

    def run(self, task: Task, config: AgentConfig) -> Episode:
        if self.sandbox is None:
            raise RuntimeError(f"{type(self).__name__} requires a sandbox.")

        env = self.build_env(task, config)
        self.write_configs(task, config, env)

        instruction = str(task.instruction).strip()
        timeout = float(task.metadata.get("agent_timeout", self.run_timeout))
        cmd = self.build_invocation(instruction, task, config)

        try:
            stdout = self._exec_agent(cmd, timeout=timeout, env=env)
        except Exception as e:
            stdout = f"{self.name} execution failed: {e}"
            logger.warning("%s: %s", type(self).__name__, stdout)

        steps = self.parse_episode(stdout, task, config)
        trajectory = Trajectory(
            uid=config.session_uid,
            name=self.name,
            task=task.id,
            steps=steps,
            output=stdout,
        )
        return Episode(
            id=config.session_uid,
            task=task.id,
            trajectories=[trajectory],
        )

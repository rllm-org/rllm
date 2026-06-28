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
import os
import shlex
import time
import uuid
from abc import abstractmethod

from rllm.env import env_int
from rllm.sandbox.protocol import Sandbox, SandboxCommandTimeout
from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow
from rllm.types import AgentConfig, Episode, Task, TerminationReason, Trajectory, termination_reason_from_error

logger = logging.getLogger(__name__)


class BaseCliHarness(SandboxedAgentFlow):
    """Run a CLI agent (opencode, mini-swe-agent, claude-code, …) inside the sandbox.

    Subclasses fill in the install command, env var dict, optional config
    files, and the invocation. ``run`` returns ``None``; the gateway
    captures every LLM call and the engine builds the trajectory.
    """

    # Display name used by the agent registry.
    name: str = "cli"
    # The CLI process makes its LLM calls from inside the sandbox, so it needs
    # the publicly-reachable gateway URL (tunnel) on remote backends.
    llm_inside_env: bool = True
    # Default sandbox backend — overridable via class attr or kwargs.
    sandbox_backend: str = "docker"
    # Default container image. Tasks usually override via per-task images.
    image: str = "python:3.11-slim"
    # Container user the CLI runs as. ``None`` uses the image default.
    agent_user: str | None = None
    # Path inside the sandbox where the CLI's stdout is teed.
    stdout_log_path: str = "/tmp/agent-stdout.log"
    # Per-call timeouts (seconds). Tasks may override via metadata.
    install_timeout: int = env_int("RLLM_HARNESS_INSTALL_TIMEOUT_S", 600)  # set env var: export RLLM_HARNESS_INSTALL_TIMEOUT_S=xxx
    run_timeout: int = env_int("RLLM_HARNESS_RUN_TIMEOUT_S", 3600)  # set env var: export RLLM_HARNESS_RUN_TIMEOUT_S=xxx
    # When an operator sets RLLM_HARNESS_RUN_TIMEOUT_S (or --agent-timeout),
    # run_timeout is a hard CEILING on the per-task ``agent_timeout``, not just a
    # fallback — effective timeout = min(agent_timeout, run_timeout). When it is
    # unset, the task's own agent_timeout governs (so eval honors each
    # benchmark's per-task budget). Captured at import; configure() flips it on.
    run_timeout_is_cap: bool = "RLLM_HARNESS_RUN_TIMEOUT_S" in os.environ
    # Grace added to the *exec* timeout over the agent's budget so an in-sandbox
    # driver that self-limits at the budget (terminus2) has time to record its
    # outcome and exit cleanly before the backend SIGKILLs the exec. The agent
    # still only "feels" ``_effective_timeout``; this just keeps the kill from
    # racing the sentinel write.
    timeout_grace_s: int = env_int("RLLM_HARNESS_TIMEOUT_GRACE_S", 60)

    def configure(self, overrides: dict) -> dict:
        """Consume CLI overrides this harness understands, then defer to the base.

        ``agent_timeout`` (seconds) caps a single rollout's wall-clock — it's the
        timeout handed to the in-sandbox ``exec``. ``rllm eval --agent-timeout``
        routes here. Backends with a hard sandbox lifetime (e.g. Modal, see
        ``modal_backend._default_sandbox_timeout``) size it to exceed this so the
        environment can't be reaped before the agent's own timeout fires.
        """
        leftovers = super().configure(overrides)
        timeout = leftovers.pop("agent_timeout", None)
        if timeout is not None:
            self.run_timeout = int(timeout)
            self.run_timeout_is_cap = True  # an explicit --agent-timeout is a hard ceiling
        return leftovers

    # ---------------------------------------------------------------------
    # Sandbox helpers
    # ---------------------------------------------------------------------

    def _exec_agent(
        self,
        sandbox: Sandbox,
        command: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        """Run *command* in *sandbox* as the agent user, with *env* exported.

        Uses ``export`` (not the inline ``K=V cmd`` prefix), because
        invocations like ``cd /workspace && claude ...`` are compound
        commands. Bash's inline assignment only applies to the *first*
        command in the chain — ``ANTHROPIC_API_KEY`` would be set for
        ``cd`` and gone by the time the CLI runs, leaving the agent
        looking like it had no auth even though we passed it.
        """
        if env:
            exports = "; ".join(f"export {k}={shlex.quote(v)}" for k, v in env.items() if v is not None)
            command = f"{exports}; {command}"
        return sandbox.exec(command, timeout=timeout, user=self.agent_user)

    def _effective_timeout(self, task: Task) -> float:
        """The agent's wall-clock budget for this task, in seconds.

        The task's own ``agent_timeout`` applies, but an operator-set
        ``RLLM_HARNESS_RUN_TIMEOUT_S`` / ``--agent-timeout`` is a hard ceiling
        over it (``run_timeout_is_cap``). Shared by :meth:`run` (to size the
        exec timeout + classify a wall-clock kill) and by harnesses that hand
        the same budget to an in-sandbox driver.
        """
        per_task = task.metadata.get("agent_timeout")
        if per_task is None:
            return float(self.run_timeout)
        if self.run_timeout_is_cap:
            return min(float(per_task), float(self.run_timeout))
        return float(per_task)

    def _read_outcome(self, sandbox: Sandbox) -> dict | None:
        """Structured outcome an in-sandbox driver may have written.

        Returns ``{"exception_type": str|None, "message": str}`` when a driver
        recorded one (an empty ``exception_type`` means it finished cleanly), or
        ``None`` when there's no sentinel — in which case :meth:`run` falls back
        to the elapsed-vs-budget heuristic. Base CLI harnesses run an opaque
        binary and write nothing, so this is ``None``; terminus2 overrides it.
        """
        return None

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

    # Provider slugs litellm accepts as the request prefix.
    _LITELLM_PROVIDER_SLUGS = frozenset(
        {
            "openai",
            "anthropic",
            "azure",
            "azure_openai",
            "bedrock",
            "vertex_ai",
            "google",
            "gemini",
            "cohere",
            "deepseek",
            "groq",
            "mistral",
            "xai",
            "perplexity",
            "fireworks_ai",
            "together_ai",
            "anyscale",
            "deepinfra",
            "huggingface",
            "ollama",
            "replicate",
            "openrouter",
            "databricks",
        }
    )

    @classmethod
    def ensure_provider_prefix(cls, model_name: str) -> tuple[str, str, str]:
        """Return ``(provider, model_id, qualified_name)`` for *model_name*.

        Accepts bare names (``gpt-4o``), pre-qualified names
        (``openai/gpt-4o``), and HF-style identifiers
        (``Qwen/Qwen3.5-35B-A3B``). For HF-style inputs whose first
        segment isn't in ``_LITELLM_PROVIDER_SLUGS``, the org is dropped
        and the provider is re-inferred from the model id.
        """
        if "/" in model_name:
            head, rest = model_name.split("/", 1)
            if head.lower() in cls._LITELLM_PROVIDER_SLUGS:
                return head, rest, model_name
            model_id = rest
            provider = cls.infer_provider(model_id)
            return provider, model_id, f"{provider}/{model_id}"
        provider = cls.infer_provider(model_name)
        return provider, model_name, f"{provider}/{model_name}"

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

        Baked into snapshot images by ``rllm snapshot create --agent``;
        otherwise run as root by ``SandboxTaskHooks`` on cold sandboxes.
        Must be idempotent — it may run on a container where the CLI is
        already present.
        """

    @abstractmethod
    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        """Return the env vars the CLI needs (auth, base URL, model)."""

    def write_configs(
        self,
        sandbox: Sandbox,
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

    def _outcome_episode(
        self,
        task: Task,
        termination_reason: TerminationReason | None = None,
        error: dict | None = None,
    ) -> Episode:
        """Outcome Episode for this run: one empty-step Trajectory (the engine
        fills its Steps from gateway traces) tagged with the reason the harness
        observed — TIMEOUT/ERROR, or None on a clean exit (engine → ENV_DONE).
        """
        metadata: dict = {}
        if error is not None:
            metadata["error"] = error
        return Episode(
            task=task.metadata,
            termination_reason=termination_reason,
            trajectories=[Trajectory(name=self.name, steps=[])],
            metadata=metadata,
        )

    def run(self, task: Task, config: AgentConfig, *, env: Sandbox) -> Episode:
        """Exec the CLI in the sandbox; the gateway captures its LLM calls.

        Install has already run (baked into the image, or by the hook on a cold
        sandbox). Returns an outcome :class:`~rllm.types.Episode` (one empty-step
        Trajectory the engine enriches from gateway traces) tagged with the
        ``termination_reason`` this run observed: ``TIMEOUT`` (wall-clock budget),
        ``ERROR`` (sandbox/exec failure), or ``None`` on a clean exit (→ ENV_DONE).
        """
        sandbox = env
        env_vars = self.build_env(task, config)
        self.write_configs(sandbox, task, config, env_vars)

        instruction = str(task.instruction).strip()
        budget = self._effective_timeout(task)
        # Give the exec a grace window over the agent's budget: a driver that
        # self-limits at ``budget`` (terminus2) gets to record its outcome before
        # this kills it. Harnesses without a driver just get a slightly later
        # hard kill, which the elapsed backstop below still classifies as TIMEOUT.
        exec_timeout = budget + self.timeout_grace_s
        cmd = self.build_invocation(instruction, task, config)

        start = time.monotonic()
        try:
            self._exec_agent(sandbox, cmd, timeout=exec_timeout, env=env_vars)
        except SandboxCommandTimeout as e:
            # The backend killed the exec at the wall — the agent spent its budget.
            # Captured steps are still scored; TIMEOUT lets compact filtering skip it.
            logger.info("%s reached its time budget: %s", type(self).__name__, e)
            return self._outcome_episode(task, termination_reason=TerminationReason.TIMEOUT)
        except Exception as e:
            # Traces up to the failure still drive enrichment; mark ERROR.
            logger.warning("%s execution failed: %s", type(self).__name__, e)
            return self._outcome_episode(
                task,
                termination_reason=TerminationReason.ERROR,
                error={"message": str(e), "error_type": type(e).__name__},
            )
        elapsed = time.monotonic() - start

        # Prefer the driver's own verdict when it left one: it knows whether the
        # agent declared completion or hit a phase timeout (AgentTimeoutError,
        # ContextLengthExceededError, ...). A masked exit code (e.g. ``| tee``
        # swallowing the kill) can't be trusted, so the sentinel — not the exit
        # status — is authoritative when present.
        outcome = self._read_outcome(sandbox)
        if outcome is not None:
            exc_type = outcome.get("exception_type")
            if exc_type:
                reason = termination_reason_from_error(exc_type, default=TerminationReason.ERROR)
                logger.info("%s outcome: %s -> %s", type(self).__name__, exc_type, reason)
                return self._outcome_episode(
                    task,
                    termination_reason=reason,
                    error={"message": outcome.get("message", ""), "error_type": exc_type},
                )
            # Driver finished cleanly (declared done) → ENV_DONE, even past the wall.
            return self._outcome_episode(task, termination_reason=None)

        # No sentinel (opaque CLI, or the driver was killed before writing one):
        # an exec that ran essentially to the wall is a wall-clock timeout whose
        # exit code got masked — the failure mode this backstop exists to catch.
        if budget > 0 and elapsed >= budget * 0.95:
            logger.info("%s ran to its wall-clock budget (%.0fs/%.0fs) with no clean exit; marking TIMEOUT", type(self).__name__, elapsed, budget)
            return self._outcome_episode(task, termination_reason=TerminationReason.TIMEOUT)

        # Clean exit well within budget: the engine marks this ENV_DONE.
        return self._outcome_episode(task, termination_reason=None)

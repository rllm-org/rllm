"""Public per-task hooks for :class:`rllm.engine.agentflow_engine.AgentFlowEngine`.

:class:`SandboxTaskHooks` is the canonical implementation, used by
``rllm eval`` and by :class:`rllm.trainer.unified_trainer.AgentTrainer`
for sandbox-style flows and env-carrying datasets.

The central question — *does this rollout need a sandbox?* — is answered in
exactly one place, :func:`resolve_rollout_plan`, as the join of three
declared needs:

* the **flow** declares it (``needs_env`` attr — ``True`` on
  :class:`~rllm.sandbox.sandboxed_flow.SandboxedAgentFlow`, ``False`` on
  ``@rllm.rollout`` flows),
* the **evaluation policy** needs it (:class:`FromTaskEvaluation` resolving a
  ``sandbox-shell`` / ``python-hybrid`` verifier; :class:`FixedEvaluation` never does),
* the **task** declares it (``environment/`` dir or ``task_path`` metadata).

Wiring-time call sites (trainer/CLI) use :func:`scan_env_requirements` over
the datasets to decide *whether to install these hooks at all* and whether
the gateway needs a public tunnel — the same predicate, evaluated before
any per-task bind.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from rllm.gateway.tunnel import is_local_sandbox_backend
from rllm.types import Evaluator, Task

if TYPE_CHECKING:
    from rllm.engine.agentflow_engine import TaskContext
    from rllm.sandbox.protocol import Sandbox
    from rllm.sandbox.snapshot import SnapshotRegistry
    from rllm.sandbox.warm_queue import WarmQueue
    from rllm.types import AgentFlow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation policies
# ---------------------------------------------------------------------------


class FixedEvaluation:
    """Evaluation policy: one host-side evaluator scores every task.

    Host-side by definition — it never contributes an env requirement.
    """

    def __init__(self, evaluator: Evaluator):
        from rllm.eval._resolution import _adapt_legacy_evaluator

        self.evaluator = _adapt_legacy_evaluator(evaluator)

    def detect(self, task: Task) -> tuple[str, dict]:  # noqa: ARG002
        return "fixed", {}

    def resolve(self, task: Task, sandbox: Sandbox | None, kind: str, config: dict) -> Evaluator:  # noqa: ARG002
        return self.evaluator


class FromTaskEvaluation:
    """Evaluation policy: resolve a per-task verifier from the task's ``[verifier]`` config."""

    def detect(self, task: Task) -> tuple[str, dict]:
        from rllm.eval._resolution import _detect_verifier

        return _detect_verifier(task)

    def resolve(self, task: Task, sandbox: Sandbox | None, kind: str, config: dict) -> Evaluator:
        from rllm.eval._resolution import _resolve_evaluator

        return _resolve_evaluator(task, sandbox, kind, config)


EvaluationPolicy = FixedEvaluation | FromTaskEvaluation

# Verifier kinds that must run against a live sandbox.
_ENV_VERIFIER_KINDS = frozenset({"sandbox-shell", "python-hybrid"})


# ---------------------------------------------------------------------------
# The env-requirement join (one place, pure, per task)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutPlan:
    """Pure bind-time resolution for one task: env requirement + verifier choice."""

    needs_env: bool
    verifier_kind: str
    verifier_config: dict = field(default_factory=dict)


def flow_needs_env(agent_flow: Any) -> bool:
    """The flow's declared env requirement (``False`` when undeclared)."""
    return bool(getattr(agent_flow, "needs_env", False))


def task_declares_env(task_or_row: Any) -> bool:
    """True when a task (or raw dataset row) declares a sandbox environment.

    Carriers: ``task_path`` metadata (harbor-sourced rows) or an
    ``environment/`` dir next to the task/dataset (local benchmarks).
    """
    meta = getattr(task_or_row, "metadata", None)
    if meta is None and isinstance(task_or_row, dict):
        meta = task_or_row
    if isinstance(meta, dict) and meta.get("task_path"):
        return True
    if isinstance(task_or_row, Task):
        return (task_or_row.task_dir / "environment").is_dir() or (task_or_row.dataset_dir / "environment").is_dir()
    return False


# Warn once per task id, not once per rollout (the same task repeats across
# group rollouts and epochs).
_warned_no_consumer: set[str] = set()


def resolve_rollout_plan(task: Task, agent_flow: Any, evaluation: EvaluationPolicy) -> RolloutPlan:
    """Decide once whether this rollout needs a sandbox, and which verifier scores it.

    ``needs_env = flow ∨ evaluator ∨ task`` — with the no-consumer rule: a
    task-declared env that neither the flow nor the evaluator can reach is
    downgraded (with a warning) instead of provisioned for nobody.
    """
    kind, config = evaluation.detect(task)
    flow_env = flow_needs_env(agent_flow)
    eval_env = kind in _ENV_VERIFIER_KINDS
    task_env = task_declares_env(task)

    needs_env = flow_env or eval_env or task_env
    if needs_env and not flow_env and not eval_env:
        if task.id not in _warned_no_consumer:
            _warned_no_consumer.add(task.id)
            logger.warning(
                "task '%s' declares a sandbox environment, but neither the agent flow (%s) nor the verifier (kind=%s) can use one — skipping sandbox provisioning",
                task.id,
                type(agent_flow).__name__,
                kind,
            )
        needs_env = False

    return RolloutPlan(needs_env=needs_env, verifier_kind=kind, verifier_config=config)


# ---------------------------------------------------------------------------
# Wiring-time scan (trainer/CLI: install hooks? tunnel?)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvRequirements:
    """Run-level env requirements: whether any rollout may need a sandbox, and remoteness."""

    needs_env: bool
    any_remote: bool


def scan_env_requirements(agent_flow: Any, *datasets: Any, sandbox_backend: str | None = None) -> EnvRequirements:
    """Scan flow + all dataset rows for env requirements before wiring.

    ``any_remote`` drives the run-global gateway tunnel decision: when
    ``sandbox_backend`` is explicitly set it wins (it overrides per-task
    metadata at provision time); otherwise any row's ``sandbox_backend``
    metadata counts.
    """
    needs_env = flow_needs_env(agent_flow)
    any_remote = sandbox_backend is not None and not is_local_sandbox_backend(sandbox_backend)

    for ds in datasets:
        if ds is None:
            continue
        for row in ds:
            if not needs_env and task_declares_env(row):
                needs_env = True
            if sandbox_backend is None and not any_remote:
                meta = getattr(row, "metadata", None) or (row if isinstance(row, dict) else {})
                row_backend = meta.get("sandbox_backend") if isinstance(meta, dict) else None
                if row_backend and not is_local_sandbox_backend(row_backend):
                    any_remote = True
            if needs_env and (any_remote or sandbox_backend is not None):
                return EnvRequirements(needs_env=needs_env, any_remote=any_remote)

    return EnvRequirements(needs_env=needs_env, any_remote=any_remote)


# ---------------------------------------------------------------------------
# SandboxTaskHooks
# ---------------------------------------------------------------------------


class SandboxTaskHooks:
    """Per-task setup/teardown hook with sandbox lifecycle + verifier resolution.

    Args:
        evaluation: :class:`FixedEvaluation` (one host evaluator for every task) or
            :class:`FromTaskEvaluation` (per-task ``[verifier]`` resolution, the default).
        sandbox_backend: Override for the sandbox backend
            (``"docker"``, ``"local"``, ``"modal"``, ...). Falls back to
            per-task ``metadata['sandbox_backend']`` then ``"docker"``.
        use_snapshot: When True (default), boot each task from a pre-built
            environment snapshot if the local registry has one (transparent
            cold-start acceleration); otherwise always take the cold path.
    """

    def __init__(
        self,
        evaluation: EvaluationPolicy | None = None,
        sandbox_backend: str | None = None,
        use_snapshot: bool = True,
    ) -> None:
        from rllm.sandbox.snapshot import SnapshotRegistry

        self.evaluation: EvaluationPolicy = evaluation if evaluation is not None else FromTaskEvaluation()
        self.sandbox_backend = sandbox_backend
        # Optional per-run warm queue (set by run_dataset / the trainer); when
        # present, setup pops a prefetched sandbox instead of creating one inline.
        self.warm_queue: WarmQueue | None = None
        # Read-only registry, loaded once and shared across this run's tasks
        # (None disables snapshots, e.g. eval --no-snapshot).
        self._registry = SnapshotRegistry.load() if use_snapshot else None
        if self._registry is not None and sandbox_backend:
            # Best-effort run-start reconcile: drop only verified-absent local
            # records so a stale ref doesn't cost an optimistic boot. A sync
            # failure must never crash the run.
            try:
                self._registry.sync(sandbox_backend)
            except Exception:
                logger.debug("snapshot sync at run start failed — continuing", exc_info=True)

    @property
    def registry(self) -> SnapshotRegistry | None:
        """The shared snapshot registry (so a warm queue fills from the same one)."""
        return self._registry

    def setup(self, task: Task, agent_flow: AgentFlow, uid: str) -> TaskContext:
        from rllm.engine.agentflow_engine import TaskContext
        from rllm.eval._resolution import _resolve_backend, _setup_task_environment

        plan = resolve_rollout_plan(task, agent_flow, self.evaluation)

        sandbox = None
        env_backend = None
        try:
            if plan.needs_env:
                from rllm.env import env_int
                from rllm.sandbox.snapshot import get_sandbox, install_script_for

                install = install_script_for(agent_flow)
                sandbox = self.warm_queue.pop(task) if self.warm_queue is not None else get_sandbox(task, self.sandbox_backend, self._registry, install)
                env_backend = _resolve_backend(task, self.sandbox_backend)
                _setup_task_environment(task, sandbox)
                # CLI install, unless the image already contains exactly this
                # script (baked_install, recorded at snapshot boot).
                if install and getattr(sandbox, "baked_install", "") != install:
                    try:
                        sandbox.exec(install, timeout=getattr(agent_flow, "install_timeout", env_int("RLLM_HARNESS_INSTALL_TIMEOUT_S", 600)), user="root")
                    except Exception as e:
                        raise RuntimeError(f"Failed to install {getattr(agent_flow, 'name', type(agent_flow).__name__)} in sandbox: {e}") from e

            evaluator = self.evaluation.resolve(task, sandbox, plan.verifier_kind, plan.verifier_config)
        except BaseException:
            # Nothing has registered a teardown yet — close the sandbox here
            # or it leaks (and the retry path provisions another).
            if sandbox is not None:
                try:
                    sandbox.close()
                except Exception:
                    logger.exception("sandbox.close failed after setup error")
            raise

        def teardown() -> None:
            # Sandboxes are ephemeral — the hook owns this one's lifecycle and
            # closes it directly (the #616 fix); flows never hold one.
            if sandbox is None:
                return
            try:
                sandbox.close()
            except Exception:
                logger.exception("sandbox.close failed")

        return TaskContext(evaluator=evaluator, env=sandbox, env_backend=env_backend, teardown=teardown)


class FixedEvaluatorHooks:
    """Hooks that bind one evaluator to every task and provision nothing.

    What :class:`~rllm.engine.agentflow_engine.AgentFlowEngine` wraps a bare
    ``evaluator=`` in, so the engine has exactly one execution path
    (always-hooks). Routes through ``_adapt_legacy_evaluator`` so dict-style
    ``evaluate(task: dict, episode)`` evaluators keep working.
    """

    def __init__(self, evaluator: Evaluator):
        from rllm.eval._resolution import _adapt_legacy_evaluator

        self.evaluator = _adapt_legacy_evaluator(evaluator)

    def setup(self, task: Task, agent_flow: AgentFlow, uid: str) -> TaskContext:  # noqa: ARG002
        from rllm.engine.agentflow_engine import TaskContext

        return TaskContext(evaluator=self.evaluator)


# ---------------------------------------------------------------------------
# Gateway wiring helpers
# ---------------------------------------------------------------------------


def pin_gateway_host_loopback(config: DictConfig) -> DictConfig:
    """Pin ``rllm.gateway.host=127.0.0.1`` if not explicitly set, so docker containers can reach it via ``host.docker.internal``."""
    if config.rllm.get("gateway", {}).get("host"):
        return config
    return OmegaConf.merge(
        config,
        OmegaConf.create({"rllm": {"gateway": {"host": "127.0.0.1"}}}),
    )


def enable_gateway_tunnel(config: DictConfig) -> DictConfig:
    """Auto-wire ``rllm.gateway.tunnel`` when no tunnel is explicitly set.

    Callers decide *when* (sandboxes run off-host — see
    :func:`scan_env_requirements`); this helper resolves *what*. An explicit
    ``rllm.gateway.tunnel`` always wins; otherwise resolution falls to
    ``$RLLM_GATEWAY_TUNNEL`` → a running ``rllm tunnel up`` daemon → a free
    Cloudflare quick tunnel (with a warning) — see
    :func:`rllm.gateway.tunnel.resolve_auto_tunnel`.
    """
    gw = config.rllm.get("gateway", {}) or {}
    if gw.get("tunnel"):
        return config

    from rllm.gateway.tunnel import resolve_auto_tunnel

    value, warning = resolve_auto_tunnel()
    if warning:
        logger.warning(warning)
    return OmegaConf.merge(
        config,
        OmegaConf.create({"rllm": {"gateway": {"tunnel": value}}}),
    )


__all__ = [
    "EnvRequirements",
    "EvaluationPolicy",
    "FixedEvaluation",
    "FixedEvaluatorHooks",
    "FromTaskEvaluation",
    "RolloutPlan",
    "SandboxTaskHooks",
    "enable_gateway_tunnel",
    "pin_gateway_host_loopback",
    "resolve_rollout_plan",
    "scan_env_requirements",
]

"""Public per-task hooks for :class:`rllm.engine.agentflow_engine.AgentFlowEngine`.

The engine runs the same driver loop for training and eval; the
sandbox-and-verifier concerns slot in via the :class:`TaskHooks` protocol
declared in :mod:`rllm.engine.agentflow_engine`. The canonical
implementation is :class:`SandboxTaskHooks` — used by ``rllm eval`` and by
:class:`rllm.experimental.unified_trainer.AgentTrainer` when training
sandboxed harnesses on harbor task dirs.

Per rollout it:

* if an ``evaluator_override`` is provided, uses it for every task and
  only allocates a sandbox when the agent itself is a
  :class:`~rllm.sandbox.sandboxed_flow.SandboxedAgentFlow`;
* otherwise detects the per-task verifier from ``task.toml`` /
  ``dataset.toml`` and the task's filesystem layout, builds a sandbox
  if either the verifier or the agent needs one, runs the task's setup
  commands inside it, binds it to ``SandboxedAgentFlow`` instances via
  :meth:`set_sandbox`, and resolves the per-task
  :class:`~rllm.types.Evaluator`;
* hands the engine a fresh per-task copy of the agent flow (via
  :meth:`SandboxedAgentFlow.create_instance`) so parallel rollouts don't
  share ``_sandbox`` state;
* returns a :class:`TaskContext` whose ``teardown`` closure tears the
  sandbox down on success or failure.

The public name is :class:`SandboxTaskHooks`; ``EvalHooks`` is kept as
an alias for back-compat with the eval-only history.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rllm.types import Evaluator

if TYPE_CHECKING:
    from rllm.engine.agentflow_engine import TaskContext
    from rllm.types import AgentFlow, Task

logger = logging.getLogger(__name__)


class SandboxTaskHooks:
    """Per-task setup/teardown hook with sandbox + per-task verifier resolution.

    Args:
        evaluator_override: If provided, all tasks use this evaluator —
            verifier detection is skipped (the override fully dictates
            scoring; useful for catalog datasets like ``math500`` where
            ``--evaluator math_evaluator`` binds one evaluator to the
            whole run).
        sandbox_backend: Override for the sandbox backend
            (``"docker"``, ``"local"``, ``"modal"``, ...). When ``None``,
            falls back to per-task ``metadata['sandbox_backend']`` then
            ``"docker"``.
    """

    def __init__(
        self,
        evaluator_override: Evaluator | None = None,
        sandbox_backend: str | None = None,
    ) -> None:
        self.evaluator_override = evaluator_override
        self.sandbox_backend = sandbox_backend

    def setup(self, task: Task, agent_flow: AgentFlow, uid: str) -> TaskContext:
        from rllm.engine.agentflow_engine import TaskContext
        from rllm.eval._resolution import (
            _adapt_legacy_evaluator,
            _create_sandbox_for_task,
            _detect_verifier,
            _needs_sandbox,
            _resolve_evaluator,
            _setup_task_environment,
        )
        from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow

        # Skip per-task verifier detection when an override is provided.
        # The override fully dictates scoring, so we shouldn't probe the
        # task dir (Harbor task dirs contain tests/test.sh + environment/
        # that the harbor trial runs separately).
        if self.evaluator_override is not None:
            verifier_kind, verifier_config = "override", {}
            needs_sandbox = isinstance(agent_flow, SandboxedAgentFlow)
        else:
            verifier_kind, verifier_config = _detect_verifier(task)
            needs_sandbox = _needs_sandbox(task, verifier_kind) or isinstance(agent_flow, SandboxedAgentFlow)

        # Per-task agent flow instance. SandboxedAgentFlow stores the
        # active sandbox on ``self._sandbox``; with the engine-bound
        # ``self.agent_flow`` shared across parallel rollouts, calling
        # ``set_sandbox`` here would clobber whatever sandbox an in-flight
        # sibling task is using. ``create_instance`` shallow-copies the
        # flow so each task owns its own ``_sandbox`` slot.
        task_flow: AgentFlow = agent_flow
        if isinstance(agent_flow, SandboxedAgentFlow):
            task_flow = agent_flow.create_instance()

        sandbox = None
        if needs_sandbox:
            sandbox = _create_sandbox_for_task(task, self.sandbox_backend)
            _setup_task_environment(task, sandbox)
            if isinstance(task_flow, SandboxedAgentFlow):
                task_flow.set_sandbox(sandbox)
                # ``on_sandbox_ready`` predates AgentConfig threading the gateway
                # session URL, so we pass None here — callers that need config
                # access should read ``self.config`` (set by run_agent_flow).
                task_flow.on_sandbox_ready({"task_path": str(task.task_dir)}, None)

        # Resolve the evaluator (override beats per-task verifier).
        if self.evaluator_override is not None:
            evaluator = _adapt_legacy_evaluator(self.evaluator_override)
        else:
            evaluator = _resolve_evaluator(task, sandbox, verifier_kind, verifier_config)

        # Capture sandbox + per-task flow into the teardown closure so the
        # right release path runs (SandboxedAgentFlow has its own
        # `teardown_sandbox`, plain agents just close the sandbox).
        def teardown() -> None:
            if sandbox is None:
                return
            if isinstance(task_flow, SandboxedAgentFlow):
                try:
                    task_flow.teardown_sandbox()
                except Exception:
                    logger.exception("teardown_sandbox failed")
            else:
                try:
                    sandbox.close()
                except Exception:
                    logger.exception("sandbox close failed")

        # Surface ``task_flow`` only when it's a fresh per-task copy;
        # for non-sandboxed flows we hand the engine the original instance
        # by leaving ``agent_flow=None`` (engine then falls back to
        # ``self.agent_flow``).
        ctx_flow = task_flow if task_flow is not agent_flow else None
        return TaskContext(evaluator=evaluator, agent_flow=ctx_flow, teardown=teardown)


# Back-compat alias. The class predates the rename and was named after
# its first caller (``rllm eval``); training uses it too now.
EvalHooks = SandboxTaskHooks


__all__ = ["SandboxTaskHooks", "EvalHooks"]

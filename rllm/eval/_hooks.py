"""EvalHooks: per-task setup/teardown for the eval execution path.

`AgentFlowEngine` runs the same driver loop for training and eval; the only
eval-specific concerns — per-task verifier resolution, sandbox lifecycle —
slot in via the engine's :class:`TaskHooks` protocol.

This module implements that hook for the ``rllm eval`` CLI:

* If an ``evaluator_override`` is provided (CLI ``--evaluator``), use it for
  every task; sandbox only if the agent itself is a ``SandboxedAgentFlow``.
* Otherwise, detect the per-task verifier from ``task.toml`` /
  ``dataset.toml`` and the task's filesystem layout. Build a sandbox if the
  verifier or agent needs one, run the task's setup commands inside it,
  bind it to ``SandboxedAgentFlow`` instances, and resolve the
  per-task :class:`Evaluator`.

Returns a :class:`TaskContext` whose ``teardown`` closure tears down the
sandbox (whichever style the flow uses) on engine completion or failure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rllm.eval._resolution import (
    _adapt_legacy_evaluator,
    _create_sandbox_for_task,
    _detect_verifier,
    _needs_sandbox,
    _resolve_evaluator,
    _setup_task_environment,
)
from rllm.types import Evaluator

if TYPE_CHECKING:
    from rllm.experimental.engine.agent_flow_engine import TaskContext
    from rllm.types import AgentFlow, Task

logger = logging.getLogger(__name__)


class EvalHooks:
    """Per-task setup/teardown hook for ``rllm eval``.

    Args:
        evaluator_override: If provided, all tasks use this evaluator —
            verifier detection is skipped (the override fully dictates
            scoring; useful for catalog datasets like math500 where the
            CLI's ``--evaluator math_evaluator`` flag binds one evaluator
            to the whole run).
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
        from rllm.experimental.engine.agent_flow_engine import TaskContext
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

        sandbox = None
        if needs_sandbox:
            sandbox = _create_sandbox_for_task(task, self.sandbox_backend)
            _setup_task_environment(task, sandbox)
            if isinstance(agent_flow, SandboxedAgentFlow):
                agent_flow.set_sandbox(sandbox)
                # ``on_sandbox_ready`` predates AgentConfig threading the gateway
                # session URL, so we pass None here — callers that need config
                # access should read ``self.config`` (set by run_agent_flow).
                agent_flow.on_sandbox_ready({"task_path": str(task.task_dir)}, None)

        # Resolve the evaluator (override beats per-task verifier).
        if self.evaluator_override is not None:
            evaluator = _adapt_legacy_evaluator(self.evaluator_override)
        else:
            evaluator = _resolve_evaluator(task, sandbox, verifier_kind, verifier_config)

        # Capture sandbox + agent_flow into the teardown closure so the
        # right release path runs (SandboxedAgentFlow has its own
        # `teardown_sandbox`, plain agents just close the sandbox).
        def teardown() -> None:
            if sandbox is None:
                return
            if isinstance(agent_flow, SandboxedAgentFlow):
                try:
                    agent_flow.teardown_sandbox()
                except Exception:
                    logger.exception("teardown_sandbox failed")
            else:
                try:
                    sandbox.close()
                except Exception:
                    logger.exception("sandbox close failed")

        return TaskContext(evaluator=evaluator, teardown=teardown)

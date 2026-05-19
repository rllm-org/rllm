"""Public per-task hooks for :class:`rllm.engine.agentflow_engine.AgentFlowEngine`.

:class:`SandboxTaskHooks` is the canonical implementation, used by
``rllm eval`` and by :class:`rllm.experimental.unified_trainer.AgentTrainer`
for sandbox-style harnesses on harbor task dirs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from rllm.types import Evaluator

if TYPE_CHECKING:
    from rllm.engine.agentflow_engine import TaskContext
    from rllm.types import AgentFlow, Task

logger = logging.getLogger(__name__)


class SandboxTaskHooks:
    """Per-task setup/teardown hook with sandbox + per-task verifier resolution.

    Args:
        evaluator_override: If set, used for every task and per-task
            verifier detection is skipped.
        sandbox_backend: Override for the sandbox backend
            (``"docker"``, ``"local"``, ``"modal"``, ...). Falls back to
            per-task ``metadata['sandbox_backend']`` then ``"docker"``.
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

        if self.evaluator_override is not None:
            verifier_kind, verifier_config = "override", {}
            needs_sandbox = isinstance(agent_flow, SandboxedAgentFlow)
        else:
            verifier_kind, verifier_config = _detect_verifier(task)
            needs_sandbox = _needs_sandbox(task, verifier_kind) or isinstance(agent_flow, SandboxedAgentFlow)

        # Shallow-copy SandboxedAgentFlow so parallel rollouts get their own _sandbox.
        task_flow: AgentFlow = agent_flow
        if isinstance(agent_flow, SandboxedAgentFlow):
            task_flow = agent_flow.create_instance()

        sandbox = None
        if needs_sandbox:
            sandbox = _create_sandbox_for_task(task, self.sandbox_backend)
            _setup_task_environment(task, sandbox)
            if isinstance(task_flow, SandboxedAgentFlow):
                task_flow.set_sandbox(sandbox)
                task_flow.on_sandbox_ready({"task_path": str(task.task_dir)}, None)

        if self.evaluator_override is not None:
            evaluator = _adapt_legacy_evaluator(self.evaluator_override)
        else:
            evaluator = _resolve_evaluator(task, sandbox, verifier_kind, verifier_config)

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

        ctx_flow = task_flow if task_flow is not agent_flow else None
        return TaskContext(evaluator=evaluator, agent_flow=ctx_flow, teardown=teardown)


def needs_sandbox_isolation(agent_flow: Any, train_dataset: Any, val_dataset: Any) -> bool:
    """True when ``agent_flow`` is a :class:`SandboxedAgentFlow` or any task carries ``task_path`` metadata."""
    try:
        from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow

        if isinstance(agent_flow, SandboxedAgentFlow):
            return True
    except ImportError:
        pass

    for ds in (train_dataset, val_dataset):
        if ds is None or len(ds) == 0:
            continue
        first = ds[0]
        meta = getattr(first, "metadata", None) or (first if isinstance(first, dict) else None)
        if isinstance(meta, dict) and meta.get("task_path"):
            return True
    return False


def pin_gateway_host_loopback(config: DictConfig) -> DictConfig:
    """Pin ``rllm.gateway.host=127.0.0.1`` if not explicitly set, so docker containers can reach it via ``host.docker.internal``."""
    if config.rllm.get("gateway", {}).get("host"):
        return config
    return OmegaConf.merge(
        config,
        OmegaConf.create({"rllm": {"gateway": {"host": "127.0.0.1"}}}),
    )


__all__ = [
    "SandboxTaskHooks",
    "needs_sandbox_isolation",
    "pin_gateway_host_loopback",
]

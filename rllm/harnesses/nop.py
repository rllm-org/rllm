"""No-op harness for grading an untouched sandbox."""

from __future__ import annotations

from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory


class NopHarness(SandboxedAgentFlow):
    """Make no agent-side changes before the task evaluator runs."""

    name = "nop"
    makes_llm_calls = False

    def run(self, task: Task, config: AgentConfig, *, env) -> Episode:
        step = Step(input=str(task.instruction), output="[nop] no changes applied")
        trajectory = Trajectory(
            uid=config.session_uid,
            name=self.name,
            task=task.id,
            steps=[step],
        )
        return Episode(
            id=config.session_uid,
            task=task.id,
            trajectories=[trajectory],
        )

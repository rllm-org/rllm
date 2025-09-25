from rllm.agents.agent import Episode
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class StrandsWorkflow(Workflow):
    def __init__(self, agent_cls, agent_args=None, **kwargs):
        super().__init__(**kwargs)
        self.agent_cls = agent_cls
        self.agent_args = dict(agent_args) if agent_args is not None else {}
        self.uid = None
        self.task = None

        # Create initial agent
        self.agent = self.agent_cls(**self.agent_args)
        self.register_agent(self.agent)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        # Use parent's reset method for proper workflow pooling
        self.reset(task=task, uid=uid)

        task_text = task.get("task", "No task specified")
        self.agent.reset_trajectory(task=task_text)

        result = self.agent(task_text)
        self.commit("strands_agent", self.agent, reset=False)

        if hasattr(result, "stop_reason") and result.stop_reason in ["end_turn", "stop_sequence"]:
            raise TerminationEvent(TerminationReason.ENV_DONE)

        raise TerminationEvent(TerminationReason.ENV_DONE)

    def reset(self, task: dict | None = None, uid: str | None = None):
        """Override reset to recreate fresh agent for zero contamination."""
        # Set workflow state
        self.uid = uid
        self.task = task
        self._completed_trajectories = []
        self._termination_buffer = None

        # Create completely fresh agent instance for zero state contamination
        self.agent = self.agent_cls(**self.agent_args)
        self.register_agent(self.agent)

        return None  # No environment to return

    def is_multithread_safe(self) -> bool:
        return True

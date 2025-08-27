import time
from rllm.agents.agent import Episode
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow

class TerminalWorkflow(Workflow):
    """Multi-step workflow for Terminal-Bench integration.

    This workflow wires a thin agent to a Terminal-Bench-based environment
    and iterates for up to ``max_steps`` steps or until the environment
    signals completion.

    Args:
        agent_cls: Class of the agent to instantiate.
        env_cls: Class of the environment to instantiate.
        agent_args: Optional constructor kwargs for ``agent_cls``.
        env_args: Optional constructor kwargs for ``env_cls``.
        max_steps: Maximum number of agent-environment interaction steps.
        sampling_params: Optional sampling parameters forwarded to the engine.
        **kwargs: Additional parameters forwarded to the base ``Workflow``.
    """
    def __init__(
        self,
        agent_cls,
        env_cls,
        agent_args=None,
        env_args=None,
        max_steps=50,
        global_agent_timeout_sec=600.0,
        sampling_params=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        agent_args = dict(agent_args) if agent_args is not None else {}
        env_args = dict(env_args) if env_args is not None else {}
        sampling_params = dict(sampling_params) if sampling_params is not None else {}

        self.agent = agent_cls(**agent_args)
        self.register_agent(self.agent)
        self.env = env_cls(**env_args)
        self.max_steps = max_steps
        self.sampling_params = sampling_params
        self.global_agent_timeout_sec = global_agent_timeout_sec

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute a multi-step Terminal-Bench workflow.

        Args:
            task: Task specification dictionary.
            uid: Unique identifier for this rollout.
            **kwargs: Unused; present for API compatibility.

        Returns:
            Episode: Post-processed episode when the workflow terminates.
        """

        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)
        self.agent.update_from_env(observation, 0, False, info)

        # Compute absolute deadline if a global agent timeout is configured
        deadline = None
        if self.global_agent_timeout_sec is not None:
            deadline = time.time() + float(self.global_agent_timeout_sec)

        for _ in range(self.max_steps):
            # Enforce global agent timeout before each step
            if deadline is not None and time.time() > deadline:
                await self._eval_and_terminate()
            
            # Get model response via rollout engine (delegates to TB Terminus under the hood)
            try:
                output = await self.get_model_response(self.agent)
            except Exception:
                await self._eval_and_terminate()

            action = self.agent.update_from_model(output.text)

            next_obs, reward, done, info = await self.run_in_executor(self.env.step, action)
            self.agent.update_from_env(next_obs, reward, done, info)

            if done:
                await self.run_in_executor(self.env.close)
                raise TerminationEvent(TerminationReason.ENV_DONE)
                
        # Terminal-Bench parity: always run tests once the agent loop ends
        await self._eval_and_terminate()

    async def _eval_and_terminate(self) -> None:
        """Run final evaluation, close environment, and terminate the workflow.

        Always raises ``TerminationEvent`` with ``TerminationReason.ENV_DONE``
        after attempting to evaluate and close the environment.
        """
        try:
            await self.run_in_executor(self.env._evaluate_completion_sync)
        finally:
            await self.run_in_executor(self.env.close)
        raise TerminationEvent(TerminationReason.ENV_DONE)

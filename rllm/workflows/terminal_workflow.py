import time

from rllm.agents.agent import Episode
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class TerminalWorkflow(Workflow):
    def __init__(
        self,
        agent_cls,
        env_cls,
        agent_args=None,
        env_args=None,
        max_steps=50,
        sampling_params=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Initialize mutable defaults
        agent_args = dict(agent_args) if agent_args is not None else {}
        env_args = dict(env_args) if env_args is not None else {}
        sampling_params = dict(sampling_params) if sampling_params is not None else {}

        self.agent = agent_cls(**agent_args)
        self.register_agent(self.agent)
        self.env = env_cls(**env_args)
        self.max_steps = max_steps
        self.sampling_params = sampling_params

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute a multi-step workflow"""

        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)
        prompt = observation["prompt"]
        self.agent.update_from_env(observation, 0, False, info)

        global_timeout_sec = float(self.env.max_agent_timeout_sec)
        start_time = time.monotonic()

        for _ in range(self.max_steps):
            # Global timeout check
            if (time.monotonic() - start_time) >= global_timeout_sec:
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

            # Prepare next prompt from environment observation
            prompt = next_obs.get("prompt", "")
        # Terminal-Bench parity: always run tests once the agent loop ends
        await self._eval_and_terminate()

    async def _eval_and_terminate(self) -> None:
        """Run final evaluation, close environment, and terminate the workflow."""
        try:
            await self.run_in_executor(self.env._evaluate_completion_sync)
        finally:
            await self.run_in_executor(self.env.close)
        raise TerminationEvent(TerminationReason.ENV_DONE)

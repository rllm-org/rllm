from pathlib import Path

from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.parsers.parser_factory import ParserFactory
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.terminal import Terminal

from rllm.agents.agent import Episode
from rllm.integrations.terminal_terminus_1 import RLLMModel
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class TerminalTerminusWorkflow(Workflow):
    """Run Terminus 1 with a generic rollout engine and return an Episode."""

    def __init__(
        self,
        rollout_engine,
        model_name: str,
        env_args: dict | None = None,
        max_steps: int = 50,
        global_agent_timeout_sec: float | None = 600.0,
        **kwargs,
    ):
        super().__init__(rollout_engine=rollout_engine, **kwargs)
        self.model_name = model_name
        self.env_args = dict(env_args) if env_args is not None else {}
        self.max_steps = max_steps
        self.global_agent_timeout_sec = global_agent_timeout_sec

        self.trial_handler: TrialHandler | None = None
        self.terminal: Terminal | None = None
        self.session = None
        self.parser = None
        self.terminus: RLLMModel | None = None

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """Reset, run Terminus to completion, evaluate, and package an Episode."""
        observation, info = await self.run_in_executor(self._reset_env, task=task, uid=uid)

        prompt = observation["prompt"]
        assert self.session is not None and self.terminus is not None

        trajectory, termination_reason = await self.terminus.run_agent_loop_with_engine(
            initial_prompt=prompt,
            session=self.session,
        )

        try:
            reward = await self.run_in_executor(self._evaluate_completion_sync)
        finally:
            await self.run_in_executor(self._close_env)

        episode = Episode(id=uid, task=task, is_correct=bool(reward > 0), trajectories=[trajectory])
        episode.termination_reason = termination_reason
        return episode

    async def _eval_and_terminate(self) -> None:
        try:
            await self.run_in_executor(self._evaluate_completion_sync)
        finally:
            await self.run_in_executor(self._close_env)
        raise TerminationEvent(TerminationReason.ENV_DONE)

    # ------------------------------ Sync helpers ------------------------------
    def _reset_env(self, task: dict, uid: str):
        """Create trial, start containers and session, and build initial prompt."""
        output_path = Path("/tmp/rllm_terminal_bench_output")
        output_path.mkdir(parents=True, exist_ok=True)

        task_path = Path(task.get("task_path"))
        instruction = task.get("instruction")
        task_id = task.get("task_id", "unknown")

        self.trial_handler = TrialHandler(
            trial_name=f"{task_id}.{uid}.rllm-run",
            input_path=task_path,
            output_path=output_path,
        )

        task_config = self.trial_handler.task
        self.parser = ParserFactory.get_parser(task_config.parser_name)

        self.terminal = Terminal(
            client_container_name=self.trial_handler.client_container_name,
            client_image_name=self.trial_handler.client_image_name,
            docker_compose_path=self.trial_handler.task_paths.docker_compose_path,
            docker_image_name_prefix=self.trial_handler.docker_image_name_prefix,
            sessions_logs_path=self.trial_handler.trial_paths.sessions_path,
            agent_logs_path=self.trial_handler.trial_paths.agent_logging_dir,
            no_rebuild=self.env_args.get("no_rebuild", False),
            cleanup=self.env_args.get("cleanup", True),
        )
        self.terminal.start()
        self.session = self.terminal.create_session("agent", is_active_stream=False, as_configured_user=True)

        self.terminus = RLLMModel(
            rollout_engine=self.rollout_engine,
            model_name=self.model_name,
            max_episodes=self.max_steps,
            global_agent_timeout_sec=self.global_agent_timeout_sec,
            api_base=self.env_args.get("api_base"),
        )

        initial_prompt = self.terminus.build_initial_prompt(instruction=instruction, terminal_state=self.session.capture_pane())

        observation = {"prompt": initial_prompt, "type": "initial"}
        info = {
            "task_id": task_id,
            "episode": 0,
            "max_steps": self.max_steps,
            "instruction": instruction,
        }
        return observation, info

    def _evaluate_completion_sync(self) -> float:
        """Copy tests, run them, parse output, and return a binary reward."""
        assert self.trial_handler is not None and self.terminal is not None

        # Copy tests into the container
        paths = [self.trial_handler.task_paths.run_tests_path]
        if self.trial_handler.task_paths.test_dir.exists():
            paths.append(self.trial_handler.task_paths.test_dir)
        self.terminal.copy_to_container(
            paths=paths,
            container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
        )

        # Choose session per config
        if self.trial_handler.task.run_tests_in_same_shell:
            print(1)
            test_session = self.session
        else:
            print(2)
            test_session = self.terminal.create_session("tests", is_active_stream=False, as_configured_user=False)

        # Execute tests
        test_script_path = str(DockerComposeManager.CONTAINER_TEST_DIR / "run-tests.sh")
        try:
            test_session.send_keys(
                [f"bash {test_script_path}", "Enter"],
                block=True,
                max_timeout_sec=self.trial_handler.task.max_test_timeout_sec,
            )
            test_output = test_session.capture_pane(capture_entire=True)
            parser_results = self.parser.parse(test_output)

            all_passed = parser_results and all(status == UnitTestStatus.PASSED for status in parser_results.values())
        except Exception:
            all_passed = False

        return 1.0 if all_passed else 0.0

    def _close_env(self):
        """Stop/cleanup terminal containers if present."""
        if self.terminal:
            self.terminal.stop()

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

from rllm.environments.base.base_env import BaseEnv
from rllm.integrations.terminal_terminus_1 import RLLMTerminus as Terminus

from terminal_bench.terminal.terminal import Terminal
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.agents.terminus_1 import CommandBatchResponse
from terminal_bench.parsers.parser_factory import ParserFactory
from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.handlers.trial_handler import TrialHandler
from pydantic import ValidationError

class TerminalTerminusEnv(BaseEnv):
	"""Environment bridging rLLM and Terminal-Bench's Terminus agent.

	Manages Docker/tmux Terminal-Bench sessions, builds prompts, executes
	command batches, and runs unit tests to compute rewards.

	Args:
		model_name: LLM model identifier used by Terminus.
		api_base: Optional base URL for the LLM API.
		task_path: Path to the Terminal-Bench task directory.
		instruction: Natural language instruction for the task.
		task_id: Identifier for the task instance.
		task: Optional task dictionary overriding individual parameters.
		max_episodes: Maximum number of steps before forced evaluation.
		cleanup: Whether to remove Docker artifacts on shutdown.
		no_rebuild: Skip Docker image rebuilds if True.
		logging_dir: Optional directory for logs and markers.
		max_test_timeout_sec: Maximum time to wait for tests to complete.
		**kwargs: Reserved for future configuration.
	"""
	
	def __init__(
		self,
  		model_name: str,
		api_base: str = None,
		task_path: str = None,
		instruction: str = None,
		task_id: str = "unknown",
		task: Dict[str, Any] = None,
		max_episodes: int = 50,
		cleanup: bool = True,
		no_rebuild: bool = False,
		logging_dir: str = None,
		max_test_timeout_sec: int = 120,
		**kwargs
	):
		"""Initialize Terminal-Bench environment."""
		# Handle both task dictionary and individual parameters
		if task is not None:
			self.task = task
			task_path = task.get("task_path")
			instruction = task.get("instruction")
			task_id = task.get("task_id", "unknown")
		else:
			self.task = {
				"task_path": task_path,
				"instruction": instruction,
				"task_id": task_id,
			}
		
		self.model_name = model_name
		self.api_base = api_base
		self.max_episodes = max_episodes
		self.cleanup = cleanup
		self.no_rebuild = no_rebuild
		self.logging_dir = Path(logging_dir) if logging_dir else None
		self.max_test_timeout_sec = max_test_timeout_sec
		
		# Task configuration (may be provided later via reset(task=...))
		self.task_path = Path(task_path) if task_path else None
		self.instruction = instruction
		self.task_id = task_id
		
		# Unique session ID
		self.session_id = f"{self.task_id}_{uuid.uuid4().hex[:8]}"
		
		# Terminal-Bench components (initialized in reset)
		self.terminal = None
		self.session = None
		self.terminus_agent = None
		self.trial_handler = None
		self.parser = None
		
		# Episode tracking
		self.current_episode = 0
		self.is_initialized = False
		
	def reset(self, task: Dict[str, Any] | None = None, uid: str | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
		"""Reset environment and return initial observation.

		Args:
			task: Optional task dictionary with ``task_path``, ``instruction``, ``task_id``.
			uid: Rollout identifier to namespace logs and sessions.

		Returns:
			Tuple[Dict[str, Any], Dict[str, Any]]: Initial observation and info.
		"""
		if task is not None:
			self.task = task
			self.task_path = Path(task.get("task_path"))
			self.instruction = task.get("instruction")
			self.task_id = task.get("task_id", "unknown")
			self.session_id = f"{self.task_id}_{uuid.uuid4().hex[:8]}"

		if not self.task_path:
			raise ValueError("TerminalTerminusEnv.reset requires a task with 'task_path'")

		# Initialize trial handler
		output_path = self.logging_dir or Path("/tmp/rllm_terminal_bench_logs")
		output_path.mkdir(parents=True, exist_ok=True)
		self.trial_handler = TrialHandler(
			trial_name=f"{self.task_id}.{uid}.rllm-run",
			input_path=self.task_path,
			output_path=output_path
		)
		
		task_config = self.trial_handler.task
		self.parser = ParserFactory.get_parser(task_config.parser_name)
		self.max_test_timeout_sec = task_config.max_test_timeout_sec
		self._initialize_terminal_sync()
		self.terminus_agent = Terminus(
			model_name=self.model_name,
			max_episodes=self.max_episodes,
			api_base=self.api_base
		)
		initial_prompt = self._build_initial_prompt_sync()
		
		self.current_episode = 0
		self.is_initialized = True

		observation = {
			"prompt": initial_prompt,
			"type": "initial"
		}
		info = {
			"task_id": self.task_id,
			"episode": self.current_episode,
			"max_episodes": self.max_episodes,
			"instruction": self.instruction
		}
		return observation, info

	def step(self, action) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
		"""Execute agent action and return environment response.

		Args:
			action: Raw string or object with ``action`` attribute containing the
				JSON command batch produced by the agent.

		Returns:
			Tuple[observation, reward, done, info].
		"""
		
		if not self.is_initialized:
			raise RuntimeError("Environment not initialized. Call reset() first.")
		
		# Ensure action is a raw JSON string
		if isinstance(action, str):
			action_str = action
		elif hasattr(action, "action"):
			action_str = action.action
		else:
			action_str = str(action)

		# Parse model response into command batch
		try:
			parsed_response = CommandBatchResponse.model_validate_json(action_str)
		except (json.JSONDecodeError, ValidationError) as e:
			# End trajectory if we can't parse the response
			reward, _ = self._evaluate_completion_sync()
			observation = {"prompt": "", "type": "terminal"}
			info = {
				"task_id": self.task_id,
				"episode": self.current_episode,
				"parse_error": True,
				"error_message": str(e),
				"is_task_complete": False
			}
			return observation, reward, True, info
		
		self._record_asciinema_marker_sync(parsed_response.model_dump_json())
		timeout_occurred, terminal_output = self._execute_commands(
			parsed_response.commands
		)
		
		# Determine whether to run tests now (on "done" or max episodes)
		should_run_tests = parsed_response.is_task_complete or self._check_episode_limit()
		if should_run_tests:
			reward, _ = self._evaluate_completion_sync()
			done = True
		else:
			reward = 0.0
			done = False
		

		self.current_episode += 1
		
		# Prepare next observation
		if done:
			observation = {"prompt": "", "type": "terminal"}
		else:
			observation = {
				"prompt": terminal_output,
				"type": "timeout" if timeout_occurred else "continuation"
			}
		info = {
			"task_id": self.task_id,
			"episode": self.current_episode,
			"max_episodes": self.max_episodes,
			"timeout_occurred": timeout_occurred,
			"is_task_complete": parsed_response.is_task_complete
		}
		return observation, reward, done, info

	def close(self):
		"""Clean up terminal and container resources."""
		if self.terminal:
			self.terminal.stop()

	@staticmethod
	def from_dict(env_args: Dict[str, Any]) -> "TerminalTerminusEnv":
		"""Create environment instance from dictionary configuration.

		If top-level task keys are present (``task_path``, ``instruction``,
		``task_id``), they are collected into a nested ``task`` dict.
		"""
		# Handle case where task data is passed at the top level
		if "task_path" in env_args and "instruction" in env_args and "task_id" in env_args:
			# Create task dict from individual parameters
			task = {
				"task_path": env_args.pop("task_path"),
				"instruction": env_args.pop("instruction"), 
				"task_id": env_args.pop("task_id")
			}
			env_args["task"] = task
		return TerminalTerminusEnv(**env_args)

	@staticmethod
	def is_multithread_safe() -> bool:
		"""Thread-safe via per-instance isolated containers."""
		return True

	def _initialize_terminal_sync(self):
		"""Initialize Docker container and tmux session synchronously."""
		# Create terminal interface
		self.terminal = Terminal(
			client_container_name=self.trial_handler.client_container_name,
			client_image_name=self.trial_handler.client_image_name,
			docker_compose_path=self.trial_handler.task_paths.docker_compose_path,
			docker_image_name_prefix=self.trial_handler.docker_image_name_prefix,
			sessions_logs_path=self.trial_handler.trial_paths.sessions_path,
			agent_logs_path=self.trial_handler.trial_paths.agent_logging_dir,
			no_rebuild=self.no_rebuild,
			cleanup=self.cleanup
		)
		
		# Start containers and get the container object
		self.terminal.start()
		
		# Create tmux session for agent interaction
		self.session = self.terminal.create_session(
			"agent",
			is_active_stream=False,
			as_configured_user=True
		)
	 
	def _build_initial_prompt_sync(self) -> str:
		"""Build initial prompt using Terminus template synchronously."""
		terminal_state = self.session.capture_pane()
		
		return self.terminus_agent.build_initial_prompt(
			self.instruction, 
			terminal_state
		)

	def _execute_commands(self, commands) -> Tuple[bool, str]:
		"""Execute command batch synchronously."""
		return self.terminus_agent.execute_commands(commands, self.session)
	 
	def _record_asciinema_marker_sync(self, marker_text: str):
		"""Record interaction marker for debugging synchronously."""
		if self.logging_dir and hasattr(self.session, 'get_asciinema_timestamp'):
			current_timestamp = self.session.get_asciinema_timestamp()
			marker_file = self.logging_dir / f"{self.task_id}_markers.jsonl"
			with open(marker_file, 'a') as f:
				json.dump({
					"timestamp": current_timestamp,
					"episode": self.current_episode,
					"marker": marker_text
				}, f)
				f.write('\n')
	 
	def _evaluate_completion_sync(self) -> Tuple[float, bool]:
		"""Evaluate task completion by running tests synchronously.

		Copies test artifacts into the container, executes the task's test script,
		parses the results, and returns a binary reward.

		Returns:
			Tuple[float, bool]: ``(reward, done)`` where reward is 1.0 if all tests
			pass, else 0.0; ``done`` is always True.
		"""
		# Ensure test artifacts are copied into the container under /tests
		paths = [self.trial_handler.task_paths.run_tests_path]
		if self.trial_handler.task_paths.test_dir.exists():
			paths.append(self.trial_handler.task_paths.test_dir)
		self.terminal.copy_to_container(
			paths=paths,
			container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
		)

		# Choose session according to run_tests_in_same_shell
		if self.trial_handler.task.run_tests_in_same_shell:
			test_session = self.session
		else:
			test_session = self.terminal.create_session(
				"tests", is_active_stream=False, as_configured_user=False
			)

		# Execute test script
		test_script_path = str(DockerComposeManager.CONTAINER_TEST_DIR / "run-tests.sh")
		try:
			test_session.send_keys(
				[f"bash {test_script_path}", "Enter"],
				block=True,
				max_timeout_sec=self.trial_handler.task.max_test_timeout_sec,
			)

			# Capture test output (blocking send_keys should be sufficient)
			test_output = test_session.capture_pane(capture_entire=True)

			# Parse test results using Terminal-Bench parser
			parser_results = self.parser.parse(test_output)

			# Check if all tests passed
			if parser_results and all(
				status == UnitTestStatus.PASSED for status in parser_results.values()
			):
				reward = 1.0
			else:
				reward = 0.0

		except Exception:
			reward = 0.0

		return reward, True

	def _check_episode_limit(self) -> bool:
		"""Check if episode limit reached."""
		return self.current_episode >= self.max_episodes - 1


"""
Terminal-Bench environment integration for rLLM.

This module provides the TerminalTerminusEnv class which bridges Terminal-Bench's
task execution framework with rLLM's reinforcement learning infrastructure.
"""

import json
import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import time
import subprocess

from rllm.environments.base.base_env import BaseEnv

# Terminal-Bench imports
from terminal_bench.terminal.terminal import Terminal
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from rllm.integrations.terminal_terminus_1 import RLLMTerminus as Terminus
from terminal_bench.agents.terminus_1 import CommandBatchResponse, Command
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.parsers.parser_factory import ParserFactory
from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.handlers.asciinema_handler import AsciinemaHandler
from pydantic import ValidationError
from terminal_bench.llms.lite_llm import LiteLLM


class TerminalTerminusEnv(BaseEnv):
	"""
	Environment for Terminal-Bench tasks using Terminus agent logic.
	
	Handles Docker container management, tmux sessions, command execution,
	and test evaluation while providing a clean synchronous interface to rLLM's 
	AgentExecutionEngine.
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
		enable_token_tracking: bool = True,
		max_agent_timeout_sec: int = 1800,
		max_test_timeout_sec: int = 120,
		**kwargs
	):
		"""
		Initialize Terminal-Bench environment.
		
		Args:
			task_path: Path to task directory (or None if using task dict)
			instruction: Task instruction text (or None if using task dict)
			task_id: Unique task identifier
			task: Task dictionary with task_path, instruction, task_id (optional)
			model_name: LLM model name for Terminus agent
			max_episodes: Maximum interaction episodes
			cleanup: Whether to cleanup containers after execution
			no_rebuild: Skip Docker image rebuilding
			logging_dir: Directory for agent logs
			enable_token_tracking: Track token usage for RL training
			max_agent_timeout_sec: Maximum timeout for agent actions.
			max_test_timeout_sec: Maximum timeout for test execution.
		"""
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
		self.enable_token_tracking = enable_token_tracking
		self.max_agent_timeout_sec = max_agent_timeout_sec
		self.max_test_timeout_sec = max_test_timeout_sec
		
		# Task configuration (may be provided later via reset(task=...))
		self.task_path = Path(task_path) if task_path else None
		self.instruction = instruction
		self.task_id = task_id
		
		# Generate unique session ID for container isolation
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
		
		# Token tracking
		self.total_input_tokens = 0
		self.total_output_tokens = 0
		self.episode_tokens = []
		
	def reset(self, task: Dict[str, Any] | None = None, uid: str | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
		"""
		Reset environment and return initial observation.
		"""
		# Update task if provided (for MultiTurnWorkflow compatibility)
		if task is not None:
			self.task = task
			self.task_path = Path(task.get("task_path"))
			self.instruction = task.get("instruction")
			self.task_id = task.get("task_id", "unknown")
			# Regenerate session id for isolation per reset
			self.session_id = f"{self.task_id}_{uuid.uuid4().hex[:8]}"

		# Validate task_path is set
		if not self.task_path:
			raise ValueError("TerminalTerminusEnv.reset requires a task with 'task_path'")

		# Initialize trial handler for task management
		# Always provide an output path to avoid environment variable issues
		output_path = self.logging_dir or Path("/tmp/rllm_terminal_bench_logs")
		output_path.mkdir(parents=True, exist_ok=True)
		
		self.trial_handler = TrialHandler(
			trial_name=f"{self.task_id}.{uid}.rllm-run",
			input_path=self.task_path,
			output_path=output_path
		)
		
		# Initialize parser for test result evaluation and propagate per-task timeouts
		task_config = self.trial_handler.task
		self.parser = ParserFactory.get_parser(task_config.parser_name)
		
		self.max_agent_timeout_sec = task_config.max_agent_timeout_sec
		self.max_test_timeout_sec = task_config.max_test_timeout_sec

		# Initialize terminal synchronously
		self._initialize_terminal_sync()
		
		# Initialize Terminus agent
		self.terminus_agent = Terminus(
			model_name=self.model_name,
			max_episodes=self.max_episodes,
			api_base=self.api_base
		)
		
		# Build initial prompt
		initial_prompt = self._build_initial_prompt_sync()
		
		# Reset episode and token counters
		self.current_episode = 0
		self.total_input_tokens = 0
		self.total_output_tokens = 0
		self.episode_tokens = []
		self.is_initialized = True

		# Initialize a LiteLLM instance for exact token counting parity
		self._token_llm = LiteLLM(model_name=self.model_name)

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
		
		print(f"observation: {observation}")
		print(f"info: {info}")
		
		return observation, info

	def step(self, action) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
		"""
		Execute agent action and return environment response.
		"""
		
		print(f"action: {action}")

		if not self.is_initialized:
			raise RuntimeError("Environment not initialized. Call reset() first.")
		
		# Ensure action is a raw JSON string (extract from Action-like object if provided)
		if isinstance(action, str):
			action_str = action
		elif hasattr(action, "action"):
			action_str = action.action
		else:
			action_str = str(action)

		# Track output tokens (exact, matching TB semantics)
		if self.enable_token_tracking:
			self.total_output_tokens += self._token_llm.count_tokens([
				{"role": "assistant", "content": action_str}
			])
			
		
		# Parse model response into command batch
		try:
			parsed_response = CommandBatchResponse.model_validate_json(action_str)
		except (json.JSONDecodeError, ValidationError) as e:
			observation, reward, done, info = self._handle_parse_error_sync(str(e))
			# If we are done due to episode cap, run tests like TB harness does
			if done:
				reward, _terminated = self._evaluate_completion_sync()
				observation = {"prompt": "", "type": "terminal"}
				info["is_task_complete"] = False
			return observation, reward, done, info
		
		# Record interaction for debugging
		self._record_asciinema_marker_sync(parsed_response.model_dump_json())
		
		# Execute commands in terminal
		timeout_occurred, terminal_output = self._execute_commands_sync(
			parsed_response.commands
		)
		
		# Determine whether to run tests now
		truncated = self._check_episode_limit()
		should_run_tests = parsed_response.is_task_complete or truncated

		if should_run_tests:
			reward, terminated = self._evaluate_completion_sync()
			done = True
		else:
			reward = 0.0
			terminated = False
			done = False
		
		# Increment episode counter
		self.current_episode += 1
		
		# Track input tokens for next prompt (exact, matching TB semantics)
		if self.enable_token_tracking and not done:
			self.total_input_tokens += self._token_llm.count_tokens([
				{"role": "user", "content": terminal_output}
			])
		
		# Store episode token counts
		self.episode_tokens.append({
			"episode": self.current_episode,
			"input_tokens": self.total_input_tokens,
			"output_tokens": self.total_output_tokens
		})
		
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
			"is_task_complete": parsed_response.is_task_complete,
			"total_input_tokens": self.total_input_tokens,
			"total_output_tokens": self.total_output_tokens,
			"episode_tokens": self.episode_tokens[-1] if self.episode_tokens else None
		}
		
		return observation, reward, done, info

	def close(self):
		"""
		Clean up terminal and container resources.
		"""
		# Rely on Terminal.stop() -> DockerComposeManager.stop() to perform cleanup
		if self.terminal:
			self.terminal.stop()
		
	@staticmethod
	def from_dict(env_args: Dict[str, Any]) -> "TerminalTerminusEnv":
		"""Create environment instance from dictionary configuration."""
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
		"""
		Terminal-Bench environment is thread-safe using unique session IDs.
		Each environment instance gets isolated Docker containers.
		"""
		return True
	 
	# Private synchronous helper methods
	 
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

	def _execute_commands_sync(self, commands) -> Tuple[bool, str]:
		"""Execute command batch synchronously."""
		# Cap timeout
		for command in commands:
			command.timeout_sec = min(command.timeout_sec, self.max_agent_timeout_sec)
		return self.terminus_agent.execute_commands(commands, self.session)
	 
	def _record_asciinema_marker_sync(self, marker_text: str):
		"""Record interaction marker for debugging synchronously."""
		if hasattr(self.session, 'get_asciinema_timestamp'):
			current_timestamp = self.session.get_asciinema_timestamp()
			# Log marker with timestamp
			if self.logging_dir:
				marker_file = self.logging_dir / f"{self.task_id}_markers.jsonl"
				with open(marker_file, 'a') as f:
					json.dump({
						"timestamp": current_timestamp,
						"episode": self.current_episode,
						"marker": marker_text
					}, f)
					f.write('\n')
	 
	def _evaluate_completion_sync(self) -> Tuple[float, bool]:
		"""
		Evaluate task completion by running tests synchronously.
		
		Returns:
			Tuple of (reward, done) where reward is 1.0 for success, 0.0 for failure
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

		except Exception as e:
			print(f"Test execution failed: {e}")
			reward = 0.0

		return reward, True  # Always done after evaluation
	 
	def _handle_parse_error_sync(self, error_msg: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
		"""Handle JSON parsing errors from model response."""
		# Format error as continuation prompt so agent can recover
		error_prompt = (
			f"Error parsing your response: {error_msg}\n\n"
			f"Please provide a valid JSON response following this schema:\n"
			f"{self.terminus_agent._response_schema}\n\n"
			f"Remember to include all required fields: state_analysis, explanation, commands, and is_task_complete."
		)
		
		observation = {"prompt": error_prompt, "type": "error"}
		
		reward = 0.0
		self.current_episode += 1  # Count this as an episode
		done = self._check_episode_limit()

		# Track input tokens for the error prompt (exact parity)
		if self.enable_token_tracking and not done:
			self.total_input_tokens += self._token_llm.count_tokens([
				{"role": "user", "content": error_prompt}
			])
		
		info = {
			"task_id": self.task_id,
			"episode": self.current_episode,
			"parse_error": True,
			"error_message": error_msg,
			"total_input_tokens": self.total_input_tokens,
			"total_output_tokens": self.total_output_tokens
		}
		
		return observation, reward, done, info

	def _check_episode_limit(self) -> bool:
		"""Check if episode limit reached."""
		return self.current_episode >= self.max_episodes - 1


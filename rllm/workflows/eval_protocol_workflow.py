import asyncio
import importlib
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import eval_protocol
from eval_protocol.models import EvaluationRow, InputMetadata, Message
from eval_protocol.pytest.default_mcp_gym_rollout_processor import MCPGymRolloutProcessor
from eval_protocol.pytest.evaluation_test_utils import build_rollout_processor_config

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.openai_engine import OpenAIEngine
from rllm.workflows.workflow import Workflow


class EvalProtocolWorkflow(Workflow):
    """
    A generic workflow that runs eval-protocol rollouts and evaluations, then converts
    results into rllm Episodes/Trajectories.

    Required workflow_args:
      - env_path: module path to an eval function (e.g., "eval_protocol.benchmarks.test_frozen_lake")

    Optional workflow_args:
      - lite_llm_prefix: prefix for model id (default: "fireworks_ai/")
      - steps, temperature, max_tokens: generation/rollout params
    """

    def __init__(self, rollout_engine: OpenAIEngine, lite_llm_prefix: str = "fireworks_ai/", temperature: float = 1.0, max_tokens: int = 4096, env_path: str = "", steps: int = 30, **kwargs):
        super().__init__(rollout_engine, **kwargs)

        self._rollout_processor_semaphore = asyncio.Semaphore(1)
        self._lite_llm_prefix = lite_llm_prefix
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._steps = steps

        if env_path == "":
            raise ValueError("Workflow arg 'env_path' is required (e.g., 'eval_protocol.benchmarks.test_frozen_lake')")
        self._env_module_path = env_path

        # Resolve evaluation function and rollout hints from env module
        self._eval_func: Callable[[EvaluationRow], Any] | None = None
        try:
            module = importlib.import_module(self._env_module_path)
        except Exception as e:
            raise ImportError(f"Failed to import env module '{self._env_module_path}': {e}") from e

        candidate_tests = [obj for _, obj in inspect.getmembers(module) if callable(obj) and hasattr(obj, "__ep_params__")]
        if not candidate_tests:
            raise ValueError(f"No evaluation tests found in '{self._env_module_path}'.")

        # Use the decorated evaluation function directly
        self._eval_func = candidate_tests[0]
        self._ep_params: dict[str, Any] = getattr(self._eval_func, "__ep_params__", {})

        self._server_script_path = self._ep_params.get("server_script_path")
        self._mcp_config_path = self._ep_params.get("mcp_config_path")
        self._rollout_processor_kwargs = self._ep_params.get("rollout_processor_kwargs") or {}
        self._mode = self._ep_params.get("mode")

        self.rollout_processor = self._ep_params.get("rollout_processor")

        assert self.rollout_processor is not None

        # Decide rollout processor; prefer the instance provided by the decorator
        if isinstance(self.rollout_processor, MCPGymRolloutProcessor):
            if self._server_script_path is None:
                raise ValueError("server_script_path is required for MCPGymRolloutProcessor")

            eval_protocol_path = Path(eval_protocol.__file__).parent.parent
            server_script_path = Path(self._server_script_path)
            self._server_script_path = eval_protocol_path / server_script_path

    async def run(self, task: dict[str, Any], uid: str, **kwargs) -> Episode:
        self.reset(task=task, uid=uid)

        eval_row = self._task_to_evaluation_row(task)

        rollout_processor_config = build_rollout_processor_config(
            rollout_processor=self.rollout_processor,
            model=f"{self._lite_llm_prefix}{getattr(self.rollout_engine, 'model', '')}",
            semaphore=self._rollout_processor_semaphore,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            steps=self._steps,
            mcp_config_path=self._mcp_config_path or "",
            server_script_path=str(self._server_script_path) if self._server_script_path else None,
            rollout_processor_kwargs=self._rollout_processor_kwargs,
            start_server=True,
            server_mode="shared",
        )

        try:
            tasks = self.rollout_processor([eval_row], rollout_processor_config)
            if not tasks:
                raise ValueError("RolloutProcessor returned no tasks")

            result_row: EvaluationRow = await tasks[0]

            evaluated_row: EvaluationRow = await self._eval_func(result_row)  # pyright: ignore[reportOptionalCall]

            episode = self._evaluation_row_to_episode(evaluated_row, task, uid)

            return episode

        except Exception as e:
            # Gracefully handle failures - return a failed Episode instead of crashing
            print(f"⚠️  Task {uid} failed: {e}")

            failed_episode = Episode(
                id=uid,
                task=task,
                is_correct=False,
                trajectories=[],
                metrics={"evaluation_reward": 0.0, "error": str(e)},
            )
            return failed_episode

    def _task_to_evaluation_row(self, task: dict[str, Any]) -> EvaluationRow:
        # Default mapping mirrors frozen_lake format; customize as needed per task
        return EvaluationRow(
            messages=[Message(role="system", content=task.get("system_prompt", ""))],
            input_metadata=InputMetadata(
                row_id=task.get("id"),
                dataset_info={
                    "environment_context": task.get("environment_context", {}),
                    "user_prompt_template": task.get("user_prompt_template", "{observation}"),
                },
            ),
        )

    def _evaluation_row_to_episode(self, row: EvaluationRow, task: dict[str, Any], uid: str) -> Episode:
        # Extract reward and metrics from evaluation_result
        if row.evaluation_result is None:
            raise ValueError("Evaluation function did not return a result")

        reward = row.evaluation_result.score
        reward_info = row.evaluation_result.metrics or {}

        def msg_to_dict(msg: Message) -> dict:
            """Convert eval_protocol Message to chat completion dict."""
            d = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            if msg.name:
                d["name"] = msg.name
            return d

        trajectory = Trajectory()
        all_messages = []

        for msg in row.messages:
            msg_dict = msg_to_dict(msg)
            all_messages.append(msg_dict)

            # Create Step with only observation and chat_completions for user or tool message
            if msg.role in ["user", "tool"]:
                new_step = Step(observation=str(msg.content or ""), chat_completions=all_messages.copy())
                trajectory.steps.append(new_step)

            # Create new Step with action/response for assistant message
            elif msg.role == "assistant":
                # Extract action: tool calls if present, otherwise message content
                action_data = msg_dict.get("tool_calls") if msg.tool_calls else str(msg.content or "")

                new_step = Step(
                    model_response=str(msg.content) if msg.content else "",
                    action=action_data,
                    chat_completions=all_messages.copy(),
                )
                trajectory.steps.append(new_step)

        # Assign final reward to the last step (sparse reward)
        if trajectory.steps:
            trajectory.steps[-1].reward = reward
            trajectory.steps[-1].info = reward_info

        trajectory.reward = reward
        trajectory.task = task

        # Create episode
        episode = Episode(
            id=uid,
            task=task,
            is_correct=(reward == 1.0),
            trajectories=[trajectory],
            metrics={"evaluation_reward": reward, **reward_info},
        )

        return episode

    def cleanup(self) -> None:
        if hasattr(self, "rollout_processor") and isinstance(self.rollout_processor, MCPGymRolloutProcessor):
            self.rollout_processor.cleanup()
